from src.data_cleaning.controller import load_opportunities


def get_opportunity_from_id(opportunity_id: str):
    opportunities = load_opportunities(drop_numero_opportunite=False)

    features = opportunities.drop(
        ["Label", "opportunity_age"],
        axis=1,
    )

    row = features[features["N° Opportunité"] == opportunity_id]

    return row.drop(["N° Opportunité"], axis=1)
