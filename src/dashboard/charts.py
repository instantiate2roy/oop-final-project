import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .dashboard_data import DashboardView


def dashboard_figure(view: DashboardView) -> Figure:
    """The dashboard's nine charts for one view; used on screen and in the PDF."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    (par, buckets, tiers), (profiles, treatments, promises), (reach, projection, write_offs) = axes

    trend = view.par_trend
    par.plot(trend["business_date"], trend["PAR30 %"], marker="o", label="PAR30")
    par.plot(trend["business_date"], trend["PAR90 %"], marker="o", label="PAR90")
    par.axvline(view.business_date, color="grey", linestyle=":", linewidth=1)
    par.set(title="Portfolio at risk (% of balance)", ylabel="%")
    par.legend()
    par.tick_params(axis="x", rotation=45)

    if len(view.buckets):
        x = range(len(view.buckets))
        buckets.bar([i - 0.2 for i in x], view.buckets["loans_by_dpd"], width=0.4, label="by DPD")
        buckets.bar([i + 0.2 for i in x], view.buckets["loans_by_mia"], width=0.4, label="by MIA")
        buckets.set_xticks(list(x), view.buckets["bucket"])
        buckets.legend()
    buckets.set(title="Loans by bucket", ylabel="loans")

    if len(view.tiers):
        tiers.bar(view.tiers["tier"], view.tiers["balance"] / 1e6, label="balance")
        tiers.bar(view.tiers["tier"], view.tiers["expected_loss"] / 1e6, label="expected loss")
        tiers.legend()
    tiers.set(title="Balance by risk tier (A = lowest PD)", ylabel="UGX million")

    if len(view.profiles):
        profiles.barh(view.profiles["profile"], view.profiles["balance"] / 1e6)
    profiles.set(title="Balance by profile", xlabel="UGX million")

    if len(view.treatments):
        treatments.bar(view.treatments["treatment_type_id"], view.treatments["started"])
    treatments.set(title="Treatments started in the period", ylabel="treatments")

    if len(view.promises):
        promises.bar(view.promises["status"], view.promises["promises"])
    promises.set(title=f"Promises due in the period (keep rate {view.summary['promise keep rate %']}%)",
                 ylabel="promises")

    reach.bar(view.reachability["band"].astype(str), view.reachability["loans"])
    reach.set(title="Reachability (last 90 days)", ylabel="loans")
    reach.tick_params(axis="x", rotation=20)

    if len(view.projection):
        shown = view.projection.drop(columns="SETTLED") / 1e6
        projection.stackplot(shown.index, shown.T.to_numpy(), labels=list(shown.columns))
        projection.legend(loc="upper left", fontsize=7)
    else:
        projection.text(0.5, 0.5, "Shown for the latest business date", ha="center", va="center")
    projection.set(title="Projected balance of today's loans (not yet settled)", xlabel="months ahead",
                   ylabel="UGX million")

    if len(view.write_offs):
        write_offs.bar(view.write_offs["months ahead"].astype(str), view.write_offs["written off (UGX)"] / 1e6)
    write_offs.set(title="Projected new write-offs", xlabel="months ahead", ylabel="UGX million")

    fig.suptitle(f"Business date {view.business_date:%Y-%m-%d}", fontsize=13)
    fig.tight_layout()
    return fig
