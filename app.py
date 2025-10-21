def plot_opportunity_timeline(opp_row, bands_df, target_date, now_dt):
    fig = go.Figure()
    y_positions = {stage: i for i, stage in enumerate(bands_df["Stage"].unique()[::-1], start=1)}
    band_height = 0.6

    # P10–P90 rectangles + median line
    for stage, yi in y_positions.items():
        q10 = bands_df[(bands_df.Stage == stage) & (bands_df.Quantile == 0.1)]
        q50 = bands_df[(bands_df.Stage == stage) & (bands_df.Quantile == 0.5)]
        q90 = bands_df[(bands_df.Stage == stage) & (bands_df.Quantile == 0.9)]
        if q10.empty or q90.empty:
            continue

        x0 = q10.iloc[0]["Start"]
        x1 = q90.iloc[0]["End"]
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=yi - band_height/2, y1=yi + band_height/2,
            opacity=0.25, line=dict(width=0)
        )

        if not q50.empty:
            fig.add_trace(go.Scatter(
                x=[q50.iloc[0]["Start"], q50.iloc[0]["End"]],
                y=[yi, yi],
                mode="lines",
                name=f"{stage} median",
                hovertemplate=f"<b>{stage}</b><br>Median: %{{x|%Y-%m-%d %H:%M}}<extra></extra>",
                showlegend=False
            ))

    # Axes and reference lines
    fig.add_vline(x=now_dt, line_width=2, line_dash="dash",
                  annotation_text="Today", annotation_position="top left")
    if pd.notna(target_date):
        fig.add_vline(x=target_date, line_width=2, line_dash="dot",
                      annotation_text="Target Close", annotation_position="top right")

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(y_positions.values()),
        ticktext=list(y_positions.keys()),
        title_text="Stage"
    )
    fig.update_layout(
        title=f"Opportunity Timeline: {opp_row['OpportunityName']} ({opp_row['OpportunityId']})",
        xaxis_title="Date / Time",
        height=max(420, 40 * len(y_positions)),
        margin=dict(l=60, r=20, t=60, b=40)
    )
    return fig

