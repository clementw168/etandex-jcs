import pandas as pd

from src.data_cleaning.utils import to_binary_one_hot, to_one_hot


def import_opportunities() -> pd.DataFrame:
    df = pd.read_csv("data/Opportunites.csv", sep=";", encoding="ISO-8859-1")
    df = df[:-7]

    return df


def load_opportunities() -> pd.DataFrame:
    df = import_opportunities()

    keep_cols = [
        "Montant",
        "Origine de l'opportunité",
        "Opération",
        "Domaine",
        "Position",
        "Type du compte",
        "Étape",
        "Date de dernière modification",
    ]

    binary_one_hot_cols = ["Opération", "Domaine", "Position"]

    one_hot_cols = [
        "Origine de l'opportunité",
        "Type du compte",
    ]

    df = df[keep_cols]

    # 'Montant' to float

    df["Montant"] = df["Montant"].str.replace(",", ".").astype(float)

    # To one hot

    df = to_binary_one_hot(df, binary_one_hot_cols)

    df = to_one_hot(df, one_hot_cols)

    # Drop NaN in 'Montant'

    df = df.dropna(subset=["Montant"])

    # Drop negative values in 'Montant'

    df = df[df["Montant"] > 0]

    df["Label"] = None

    # if 'Étape' in ['6- Gagnée', '9- Gagnée archivée'] then 'Label' = 1
    # If 'Étape' in ['7- Perdue', '8- Perdue archivée'] then 'Label' = 0

    df.loc[df["Étape"].isin(["6- Gagnée", "9- Gagnée archivée"]), "Label"] = 1
    df.loc[df["Étape"].isin(["7- Perdue", "8- Perdue archivée"]), "Label"] = 0

    df["jours depuis dernière modification"] = (
        pd.to_datetime("today")
        - pd.to_datetime(df["Date de dernière modification"], format="%d/%m/%Y")
    ).dt.days

    # if 'Label' is NaN and 'jours depuis dernière modification' > 2*365 then 'Label' = 0
    df.loc[
        (df["Label"].isna()) & (df["jours depuis dernière modification"] > 2 * 365),
        "Label",
    ] = 0

    # Drop NaN in 'Label'

    df = df.dropna(subset=["Label"])

    # Drop 'Étape', 'Date de dernière modification' and 'jours depuis dernière modification'

    df = df.drop(
        [
            "Étape",
            "Date de dernière modification",
            "jours depuis dernière modification",
        ],
        axis=1,
    )

    return df
