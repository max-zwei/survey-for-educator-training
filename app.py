"""Aggregate survey dashboard. Run: streamlit run app.py

Install dependencies with: python -m pip install -r requirements.txt
The two CSV exports are resolved beside this file, regardless of working directory.
"""

from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st


INPUT_DIR = Path(__file__).resolve().parent
SINGLE = (2, 3, 4, 5, 7, 10, 11, 27)
CHECKBOX = (6, 8, 13, 14, 15, 16, 17, 19, 20, 24, 26)
RATING = (22, 25)
FREE_TEXT = (9, 12, 18, 21, 23, 29)
QUESTION_IDS = tuple(sorted((1,) + SINGLE + CHECKBOX + RATING))
# Compact structural checks, not a duplicate of the full export schema.
OPTION_COUNTS = dict(zip(CHECKBOX, (6, 5, 6, 12, 10, 10, 6, 12, 7, 10, 12)))
CONSENT_LABELS = (
    "Ich habe die Informationen gelesen und willige freiwillig ein, dass LOA e.V. "
    "meine Antworten zu den genannten Zwecken verarbeitet.",
    "Ich willige zusätzlich ein, dass nicht rückführbare, anonymisierte Zitate aus "
    "meinen Antworten verwendet werden. Diese Einwilligung ist freiwillig und keine "
    "Voraussetzung für die Teilnahme.",
)
# Observed export vocabulary, as in the notebook; not all possible survey choices.
ALLOWED_SINGLE = {
    2: ["Divers", "m", "w"],
    3: ["Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Hamburg", "Hessen",
        "Mecklenburg-Vorpommern", "Nordrhein-Westfalen", "Sachsen"],
    4: ["5 - 20 Jahre", "mehr als 20 Jahre", "weniger als 5 Jahre"],
    5: ["Andere", "Coaches / Referenten", "Freischaffende mit pädagogischer Ausbildung",
        "Freischaffende ohne pädagogische Ausbildung", "Lehrkräfte", "Sozialpädagogen"],
    7: ["1.000 - 5.000 € Jahr", "weniger als 1.000 € / Jahr"],
    10: ["1 – 8 h", "8 – 40 h", "mehr als 40 h", "weniger als 1 h"],
    11: ["Gleich", "Mehr", "Weniger"],
    27: ["1 - 8 h", "mehr als 8 h", "weniger als 1 h"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_export(path):
    """Load every column as text; normalize surrounding whitespace and blanks."""
    try:
        frame = pl.read_csv(path, infer_schema=False)
    except (OSError, UnicodeError, pl.exceptions.PolarsError) as exc:
        # Parser errors can contain private respondent values; do not display them.
        raise ValueError(f"Could not read {path.name}. Check the local CSV export.") from exc
    return frame.select(pl.all().str.strip_chars().replace("", None))


def question_columns(columns):
    """Discover question columns while enforcing the recognized survey structure."""
    by_q = {}
    for column in columns:
        match = re.fullmatch(r"Question (\d+) - (.+)", column)
        if match:
            by_q.setdefault(int(match[1]), []).append(column)
        else:
            require(not column.startswith("Question"), "Malformed question column header.")
    require(set(by_q) == set(QUESTION_IDS + FREE_TEXT),
            "Survey question IDs changed; review the export before analysis.")
    require({"ID", "completed"}.issubset(columns), "Required export metadata is missing.")
    for q in SINGLE + (25,):
        require(len(by_q[q]) == 1, f"Q{q}: expected one response column.")
    for q, count in {1: 2, 22: 4, **OPTION_COUNTS}.items():
        require(len(by_q[q]) == count, f"Q{q}: unexpected number of item columns.")
        parts = [column.split(" - ", 2) for column in by_q[q]]
        require(all(len(part) == 3 and part[1].strip() and part[2].strip() for part in parts),
                f"Q{q}: missing question or item label.")
        require(len({part[1] for part in parts}) == 1,
                f"Q{q}: inconsistent question titles.")
        require(len({part[2] for part in parts}) == count,
                f"Q{q}: duplicate item labels.")
    require(tuple(column.split(" - ", 2)[2] for column in by_q[1]) == CONSENT_LABELS,
            "Q1: processing and optional quotation consent columns changed or were reordered.")
    return by_q


def same_ids(left, right):
    return left.select("ID").sort("ID").equals(right.select("ID").sort("ID"))


def validate_exports(source, target_source):
    """Validate all rows, including ineligible rows, without exposing their values."""
    require(source.columns == target_source.columns,
            "The two exports must have identical column schemas and order.")
    by_q = question_columns(source.columns)
    for name, frame in (("Overall export", source), ("Target export", target_source)):
        require(frame["ID"].null_count() == 0 and frame["ID"].n_unique() == frame.height,
                f"{name}: IDs must be nonempty and unique.")
        require(frame.filter(pl.col("completed").is_null()
                             | ~pl.col("completed").is_in(["0", "1"])).is_empty(),
                f"{name}: completion must be 0 or 1, without blanks.")
        for q in (1,) + CHECKBOX:
            for column in by_q[q]:
                label = column.split(" - ", 2)[2]
                require(frame.filter(pl.col(column).is_not_null()
                                     & (pl.col(column) != label)).is_empty(),
                        f"{name}: Q{q} checkbox values must be the option label or blank.")
        for q in RATING:
            for column in by_q[q]:
                require(frame.filter(pl.col(column).is_not_null()
                                     & ~pl.col(column).is_in(["1", "2", "3", "4"])).is_empty(),
                        f"{name}: Q{q} ratings must be codes 1–4 or blank.")
        for q in SINGLE:
            column = by_q[q][0]
            require(frame.filter(pl.col(column).is_not_null()
                                 & ~pl.col(column).is_in(ALLOWED_SINGLE[q])).is_empty(),
                    f"{name}: Q{q} contains an unrecognized category; review the export vocabulary.")
    require(target_source.join(source.select("ID"), on="ID", how="anti").is_empty(),
            "Target export contains respondents absent from the overall export.")
    shared = source.join(target_source.select("ID"), on="ID", how="semi").sort("ID")
    require(shared.equals(target_source.sort("ID")),
            "Shared rows differ after whitespace/blank normalization; reconcile the exports.")
    require(same_ids(source.filter(pl.col(by_q[25][0]) == "4"), target_source),
            "Target export must be exactly the Q25=4 subset of the entire overall export.")
    return by_q


def eligible_expr(by_q):
    return (pl.col("completed") == "1") & pl.col(by_q[1][0]).is_not_null()


def load_data(directory=INPUT_DIR):
    """Return normalized source groups, eligible groups, and question columns."""
    source = load_export(Path(directory) / "data.csv")
    target_source = load_export(Path(directory) / "target-audience.csv")
    by_q = validate_exports(source, target_source)
    sources = {"Overall": source, "Target": target_source}
    groups = {name: frame.filter(eligible_expr(by_q)) for name, frame in sources.items()}
    require(same_ids(groups["Overall"].filter(pl.col(by_q[25][0]) == "4"), groups["Target"]),
            "Eligible target membership disagrees with the Q25=4 selection rule.")
    return sources, groups, by_q


def consent_audit(sources, by_q):
    """Aggregate source-level consent, exclusion, and inclusion counts."""
    consent, quote = by_q[1]
    eligible = eligible_expr(by_q)
    return pl.concat([
        frame.select(
            pl.lit(name).alias("Group"), pl.len().alias("Source rows"),
            (pl.col("completed") == "1").sum().alias("Completed"),
            (pl.col("completed") != "1").sum().alias("Incomplete"),
            pl.col(consent).is_not_null().sum().alias("Processing consent present"),
            pl.col(consent).is_null().sum().alias("Processing consent missing"),
            ((pl.col("completed") == "1") & pl.col(consent).is_null()).sum()
            .alias("Completed without processing consent"),
            pl.col(quote).is_not_null().sum().alias("Optional quotation consent present"),
            pl.col(quote).is_null().sum().alias("Optional quotation consent missing"),
            (~eligible).sum().alias("Excluded"), eligible.sum().alias("Included"),
            (eligible & pl.col(quote).is_not_null()).sum().alias("Included with quotation consent"),
        )
        for name, frame in sources.items()
    ])


def question_title(q, by_q):
    return by_q[q][0].split(" - ", 2)[1]


def answered_expr(q, by_q, column=None):
    if q in CHECKBOX:
        return pl.any_horizontal([pl.col(c).is_not_null() for c in by_q[q]])
    return pl.col(column or by_q[q][0]).is_not_null()


def denominators(q, groups, by_q, column=None):
    """Question/item-specific eligible, answered, and missing counts."""
    answered = answered_expr(q, by_q, column)
    return pl.concat([
        frame.select(pl.lit(name).alias("Group"), pl.len().alias("Eligible"),
                     answered.sum().alias("Answered"), (~answered).sum().alias("Missing"))
        for name, frame in groups.items()
    ])


def distribution(q, groups, by_q, column=None):
    """Category counts, available-answer percentages, and Target − Overall pp."""
    column = column or by_q[q][0]
    if q in CHECKBOX:
        labels = [c.split(" - ", 2)[2] for c in by_q[q]]
    elif q in RATING:
        labels = ["1", "2", "3", "4"]
    else:
        labels = ALLOWED_SINGLE[q]
    result = pl.DataFrame({"Response": labels})
    for name, frame in groups.items():
        answered = frame.select(answered_expr(q, by_q, column).sum()).item()
        if q in CHECKBOX:
            counts = frame.select([
                pl.col(c).is_not_null().sum().alias(label)
                for c, label in zip(by_q[q], labels)
            ]).unpivot(variable_name="Response", value_name=f"{name}_n")
        else:
            counts = (frame.filter(pl.col(column).is_not_null()).group_by(column)
                      .len(name=f"{name}_n").rename({column: "Response"}))
        result = result.join(counts, on="Response", how="left", maintain_order="left").with_columns(
            pl.col(f"{name}_n").fill_null(0))
        percent = (pl.col(f"{name}_n") * 100.0 / answered if answered
                   else pl.lit(None, dtype=pl.Float64))
        result = result.with_columns(percent.alias(f"{name}_%"))
    return result.with_columns((pl.col("Target_%") - pl.col("Overall_%")).alias("Diff_pp"))


def ordinal_summary(groups, column):
    """Lower observed-category quartiles; no arithmetic averaging of ordinal codes."""
    rating = pl.col(column).cast(pl.Int64)
    return pl.concat([
        frame.select(pl.lit(name).alias("Group"),
                     rating.quantile(0.25, interpolation="lower").alias("Lower quartile"),
                     rating.quantile(0.5, interpolation="lower").alias("Lower median"),
                     rating.quantile(0.75, interpolation="lower").alias("Upper quartile"))
        for name, frame in groups.items()
    ])


def draw_distribution(table, denom):
    """Return a horizontal chart with line-aware row spacing and wrapped labels."""
    labels = [textwrap.fill(label, width=45) for label in table["Response"].to_list()]
    row_heights = [max(0.8, 0.27 * (label.count("\n") + 1) + 0.15) for label in labels]
    positions = []
    cursor = 0.0
    for height in row_heights:
        positions.append(cursor + height / 2)
        cursor += height
    fig, ax = plt.subplots(figsize=(12, max(3.4, cursor + 1.3)), layout="constrained")
    for name, offset, color in (("Overall", -0.17, "#376B92"), ("Target", 0.17, "#C36B20")):
        answered = denom.filter(pl.col("Group") == name)["Answered"].item()
        if answered:
            values = table[f"{name}_%"].to_list()
            bars = ax.barh([y + offset for y in positions], values, height=0.3,
                           color=color, label=f"{name} (answered n={answered})")
            ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3, fontsize=9)
        else:
            ax.plot([], [], color=color, linewidth=6, label=f"{name}: no data (answered n=0)")
    if not denom["Answered"].sum():
        ax.text(0.5, 0.5, "No answered responses in either group", transform=ax.transAxes,
                ha="center", va="center")
    ax.set_yticks(positions, labels, fontsize=10)
    ax.set_ylim(cursor + 0.1, -0.1)
    ax.set_xlim(0, 116)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Percent of answered responses (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig


def render_question(q, groups, by_q):
    st.header(f"Q{q} — {question_title(q, by_q)}")
    if q == 25:
        st.warning("Selection artifact: Target is defined by Q25 = 4. Its distribution is a "
                   "selection-rule check, not evidence of an independently observed preference difference.")
    if q in CHECKBOX:
        st.caption("Multiple selections allowed. At least one selection defines an answered question; "
                   "all-blank questions are missing. Percentages can sum to more than 100%.")
    if q in RATING:
        st.caption("Ordered rating codes 1–4: higher is better. Quartiles and median use the lower "
                   "observed category; codes are not averaged.")
    for column in (by_q[q] if q in RATING else [None]):
        if q == 22:
            st.subheader(column.split(" - ", 2)[2])
        denom = denominators(q, groups, by_q, column)
        table = distribution(q, groups, by_q, column)
        st.table(denom)
        fig = draw_distribution(table, denom)
        try:
            st.pyplot(fig, width="stretch")
        finally:
            plt.close(fig)
        st.table(table.with_columns(pl.col("Overall_%", "Target_%", "Diff_pp").round(1)).rename({
            "Overall_n": "Overall count", "Overall_%": "Overall %",
            "Target_n": "Target count", "Target_%": "Target %",
            "Diff_pp": "Target − Overall (pp)",
        }))
        if q in RATING:
            st.table(ordinal_summary(groups, column))
        st.caption("Percentages use the answered denominators above. Differences are Target minus "
                   "Overall in percentage points (pp); undefined percentages remain missing, not zero.")


def main():
    st.set_page_config(page_title="Survey comparison", layout="wide")
    st.title("Survey comparison: Target versus Overall")
    st.info("Only free-text questions are omitted. Q9, Q12, Q18, Q21, Q23 and Q29 are excluded. "
            "Export metadata are not survey questions; there is no Q28 in this survey export.")
    try:
        sources, groups, by_q = load_data()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except (OSError, UnicodeError, pl.exceptions.PolarsError):
        st.error("The local survey exports could not be processed. Check their structure and encoding; "
                 "respondent values are intentionally not shown.")
        st.stop()

    overall_metric, target_metric = st.columns(2)
    overall_metric.metric("Eligible Overall (ALL)", groups["Overall"].height)
    target_metric.metric("Eligible Target (Q25 = 4)", groups["Target"].height)
    st.caption("Target respondents are also included in Overall. These are overlapping descriptive "
               "cohorts, not independent groups.")
    with st.expander("Methodology and setup"):
        st.markdown("""
- **Eligibility:** completed = 1 and nonempty processing consent (the first Q1 checkbox).
  Optional quotation consent is not required; termination metadata are not a filter.
- **Normalization and validation:** surrounding whitespace is stripped; blanks become null.
  Both exports must have unique, nonempty IDs, identical schemas, identical normalized shared
  rows, recognized survey questions and expected completion, checkbox and rating encodings.
  Target must equal the Q25 = 4 subset before and after eligibility filtering.
- **Denominators:** single-choice questions and each rating item use their nonmissing responses.
  Checkbox questions are answered only if at least one option is selected. Within answered
  questions, blank options mean unselected; wholly blank questions are excluded, not all-negative.
  Routing or nonresponse may explain missingness; these cannot be distinguished from the CSV alone.
- **Distributions:** original German labels are retained. Checkbox options include never-selected
  choices. Single-choice vocabulary follows the notebook's observed export categories, including
  categories with zero eligible responses; it does not invent unobserved questionnaire options.
  Counts and percentages are shown separately; differences are Target − Overall in percentage points.
- **Ordinal summaries:** rating codes 1–4 are ordered, higher is better. Quartiles and median use
  the observed category at index floor((n − 1) × p), with p = 0.25, 0.5, 0.75. Empty items have
  missing summaries. Time, experience and budget brackets remain categories, not numeric midpoints.
- **Interpretation:** small, selected, overlapping samples do not establish representativeness or
  causality. No independent-group tests, p-values or confidence intervals are used. Q25 differences
  are imposed by the target selection rule.
- **Privacy:** aggregate outputs only; no respondent IDs, contact details or free-text answers are
  displayed. Small aggregate cells may still be sensitive; review before public sharing.
""")
        st.caption("Setup: python -m pip install -r requirements.txt · Run: streamlit run app.py. "
                   "Both CSV files are loaded beside app.py; the notebook is not executed.")

    tabs = st.tabs([f"Q{q}" for q in QUESTION_IDS])
    for q, tab in zip(QUESTION_IDS, tabs):
        with tab:
            if q == 1:
                st.header(f"Q1 — {question_title(q, by_q)}")
                st.subheader("Processing consent")
                st.write(CONSENT_LABELS[0])
                st.subheader("Optional quotation consent")
                st.write(CONSENT_LABELS[1])
                st.caption("Source-level consent audit; counts only. Included = completed with processing "
                           "consent. Excluded = incomplete + completed without processing consent. "
                           "Optional quotation consent never determines inclusion.")
                audit = consent_audit(sources, by_q)
                st.table(audit.drop("Group").transpose(
                    include_header=True, header_name="Measure", column_names=audit["Group"].to_list()))
            else:
                render_question(q, groups, by_q)


if __name__ == "__main__":
    main()
