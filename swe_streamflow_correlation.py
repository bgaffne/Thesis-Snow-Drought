"""Compare annual SNOTEL SWE with annual Twin Springs streamflow.

Run from the repository root with: python swe_streamflow_correlation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "Boise_SNOTEL_TwinSprings_exceedance.final.xlsx"
STREAMFLOW_FILE = BASE_DIR / "Data" / "twinsprings_annual_streamflow.csv"
SWE_SHEET = "Apr 1 SWE"  # Change to "Max SWE" to use the maximum-SWE sheet.


def main() -> None:
    swe_by_year = pd.read_excel(EXCEL_FILE, sheet_name=SWE_SHEET)
    streamflow_by_year = pd.read_csv(STREAMFLOW_FILE)

    swe_by_year["Water Year"] = pd.to_numeric(
        swe_by_year["Water Year"], errors="coerce"
    )
    streamflow_by_year["year"] = pd.to_numeric(
        streamflow_by_year["year"], errors="coerce"
    )
    streamflow_by_year["annual_total_acre_ft"] = pd.to_numeric(
        streamflow_by_year["annual_total_acre_ft"], errors="coerce"
    )

    swe_columns = [column for column in swe_by_year.columns if column != "Water Year"]
    for column in swe_columns:
        swe_by_year[column] = pd.to_numeric(swe_by_year[column], errors="coerce")

    flow_columns = ["year", "annual_total_acre_ft"]
    if "observed_hours" in streamflow_by_year.columns:
        flow_columns.append("observed_hours")

    annual_pairs = (
        swe_by_year.merge(
            streamflow_by_year[flow_columns],
            left_on="Water Year",
            right_on="year",
            how="inner",
            validate="one_to_one",
        )
        .drop(columns="year")
        .sort_values("Water Year")
    )

    correlation_rows = []
    for site in swe_columns:
        paired_values = annual_pairs[[site, "annual_total_acre_ft"]].dropna()
        correlation_rows.append(
            {
                "SNOTEL site": site,
                "years paired": len(paired_values),
                "Pearson r": paired_values[site].corr(
                    paired_values["annual_total_acre_ft"]
                ),
            }
        )

    correlations = pd.DataFrame(correlation_rows).sort_values(
        "Pearson r",
        key=lambda values: values.abs(),
        ascending=False,
        na_position="last",
    )

    print(f"Matched annual observations using the {SWE_SHEET!r} sheet:")
    print(annual_pairs.to_string(index=False))
    print("\nPearson correlation across matched years for each SNOTEL site:")
    print(correlations.to_string(index=False))

    plot_data = correlations.dropna(subset=["Pearson r"]).sort_values("Pearson r")
    if plot_data.empty:
        raise ValueError(
            "No SNOTEL site has enough paired, varying data to calculate a correlation."
        )

    colors = [
        "steelblue" if value >= 0 else "indianred"
        for value in plot_data["Pearson r"]
    ]
    ax = plot_data.plot.bar(
        x="SNOTEL site",
        y="Pearson r",
        color=colors,
        legend=False,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Pearson correlation (r)")
    ax.set_title(f"{SWE_SHEET} SWE vs. annual streamflow")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
