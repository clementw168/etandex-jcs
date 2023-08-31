import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data_cleaning.utils import to_binary_one_hot, to_one_hot, unique_values


def import_opportunities() -> pd.DataFrame:
    df = pd.read_csv("data/Opportunites.csv", sep=";", encoding="ISO-8859-1")
    df = df[:-7]

    return df


def drop_double_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["N° Opportunité"])


def process_montant(df: pd.DataFrame) -> pd.DataFrame:
    df["Montant"] = df["Montant"].str.replace(",", ".").astype(float)
    df = df.dropna(subset=["Montant"])
    df = df[df["Montant"] > 0]

    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
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


def create_opportunity_age(df: pd.DataFrame) -> pd.DataFrame:
    df["opportunity_age"] = (
        pd.to_datetime("today")
        - pd.to_datetime(df["Date de création"], format="%d/%m/%Y")
    ).dt.days / 365

    df = df.drop(["Date de création"], axis=1)

    return df


def load_staff(opportunities: pd.DataFrame) -> pd.DataFrame:
    staff = pd.read_csv("data/Staff.csv", sep=";", encoding="ISO-8859-1")

    staff = staff[staff["Identifiant"].isin(opportunities["Resp..1"])]

    keep_cols_staff = [
        "Identifiant",
        "fDateDeNaissance",
        "fDateEntree",
        "sDiplomeFinal",
        "sFicheDePoste",
    ]

    staff = staff[keep_cols_staff]

    one_hot_cols_staff = ["sDiplomeFinal", "sFicheDePoste"]

    staff = to_one_hot(staff, one_hot_cols_staff)

    staff["age"] = (
        pd.to_datetime("today")
        - pd.to_datetime(staff["fDateDeNaissance"], format="%d/%m/%Y")
    ).dt.days / 365

    staff["anciennete"] = (
        pd.to_datetime("today")
        - pd.to_datetime(staff["fDateEntree"], format="%d/%m/%Y")
    ).dt.days / 365

    staff = staff.drop(["fDateDeNaissance", "fDateEntree"], axis=1)

    return staff


def join_staff(df: pd.DataFrame) -> pd.DataFrame:
    # Clean 'Resp.' and 'Resp..1'
    df.loc[
        (df["Resp..1"] == "LIL") & (df["Resp."] == "Julien MILLE"), "Resp..1"
    ] = "MIL"

    staff = load_staff(df)

    df = df.join(staff.set_index("Identifiant"), on="Resp..1")

    df = df.drop(["Resp..1", "Resp."], axis=1)

    return df


def separate_code_postal(insee_com: pd.DataFrame) -> pd.DataFrame:
    """'Code Postal' can be a list of code postal separated by '/'"""
    insee_com_single = insee_com[insee_com["Code Postal"].map(lambda x: len(x) == 5)]

    insee_com_multiple = []

    for _, line in insee_com[
        insee_com["Code Postal"].map(lambda x: len(x) != 5)
    ].iterrows():
        for code_postal in line["Code Postal"].split("/"):
            item = line.copy()
            item["Code Postal"] = code_postal
            insee_com_multiple.append(item)

    insee_com_multiple = pd.DataFrame(insee_com_multiple)

    return pd.concat((insee_com_multiple, insee_com_single))


def correct_missing_postals(df: pd.DataFrame, insee_com: pd.DataFrame) -> pd.DataFrame:
    missing_postals = df.loc[
        ~df["code_postal"].isin(insee_com["Code Postal"]), "code_postal"
    ].unique()
    missing_postals = [str(x) for x in missing_postals if x != np.nan]

    insee_com_missing_postals = []

    insee_com["hash_commune"] = insee_com["Commune"].str.lower()

    manual_corrected_postals = {
        "lyon": "69001",
        "marseille": "13001",
        "paris": "75001",
        "strasbourg cedex": "67000",
        "sarreguemines": "57200",
    }

    for _, line in (
        df[df["code_postal"].isin(missing_postals)]
        .drop_duplicates(subset=["nom_ville", "code_postal"])
        .iterrows()
    ):
        if line["nom_ville"] in insee_com["hash_commune"].unique():
            sub_df = insee_com[insee_com["hash_commune"] == line["nom_ville"]]
            insee_line = sub_df.loc[sub_df["Population"].idxmax()]
            if len(insee_line.shape) > 1:
                insee_line = insee_line.iloc[0]
            insee_line["Code Postal"] = line["code_postal"]

            insee_com_missing_postals.append(insee_line)

        elif line["nom_ville"] in manual_corrected_postals:
            sub_df = insee_com[
                insee_com["Code Postal"] == manual_corrected_postals[line["nom_ville"]]
            ]
            insee_line = sub_df.loc[sub_df["Population"].idxmax()]
            if len(insee_line.shape) > 1:
                insee_line = insee_line.iloc[0]
            insee_line["Code Postal"] = line["code_postal"]

            insee_com_missing_postals.append(insee_line)

        else:
            # print(line["nom_ville"], line["code_postal"])
            ()

    insee_com_missing_postals = pd.DataFrame(insee_com_missing_postals)

    return pd.concat([insee_com, insee_com_missing_postals])


def join_insee_communes(df: pd.DataFrame) -> pd.DataFrame:
    # Correct miss filled 'Ville'
    df.loc[df["Ville"] == "*75019*", "Ville"] = "75019 PARIS-19E__ARRONDISSEMENT"

    # Separate 'Ville' in 'nom_ville' and 'code_postal' when possible
    df["nom_ville"] = df["Ville"].apply(
        lambda x: x[6:].lower() if str(x)[:5].isdigit() else np.nan
    )
    df["code_postal"] = df["Ville"].apply(
        lambda x: x[:5] if str(x)[:5].isdigit() else np.nan
    )
    df = df.copy()

    # Import 'correspondance-code-insee-code-postal.csv'
    insee_com = pd.read_csv("data/correspondance-code-insee-code-postal.csv", sep=";")

    insee_com = separate_code_postal(insee_com)

    insee_com = correct_missing_postals(df, insee_com)

    insee_com["densite"] = insee_com["Population"] / insee_com["Superficie"]
    insee_com["longitude"] = insee_com["geo_point_2d"].apply(
        lambda x: float(x.split(",")[0]) if type(x) == str else np.nan
    )
    insee_com["latitude"] = insee_com["geo_point_2d"].apply(
        lambda x: float(x.split(",")[1]) if type(x) == str else np.nan
    )

    df = pd.merge(
        df,
        insee_com.drop_duplicates(subset=["Code Postal"]),
        left_on="code_postal",
        right_on="Code Postal",
        how="left",
    )

    df = df.drop(
        [
            "Ville",
            "nom_ville",
            "code_postal",
            "Code Postal",
            "Commune",
            "Département",
            "Région",
            "Statut",
            "geo_point_2d",
            "geo_shape",
            "ID Geofla",
            "Code Commune",
            "Code Canton",
            "Code Arrondissement",
            # "Code Département",
            "Code Région",
            "hash_commune",
            "longitude",
            "latitude",
            "Superficie",
            "Population",
            "densite",
            "Altitude Moyenne",
        ],
        axis=1,
    )

    # replace 'Code Département' '2A' and '2B' by '20'
    df.loc[df["Code Département"] == "2A", "Code Département"] = "20"
    df.loc[df["Code Département"] == "2B", "Code Département"] = "20"

    df["Code Département"] = df["Code Département"].astype(float)

    return df


def join_prixm2(df: pd.DataFrame) -> pd.DataFrame:
    prixm2 = pd.read_csv("data/prixm2-communes-2017.csv", sep=",")
    prixm2 = prixm2[["INSEE_COM", "Prixm2"]]

    df = pd.merge(df, prixm2, left_on="Code INSEE", right_on="INSEE_COM", how="left")

    df = df.drop(["INSEE_COM"], axis=1)

    return df


def load_opportunities(drop_numero_opportunite: bool = True) -> pd.DataFrame:
    df = import_opportunities()

    # Drop double in 'N° Opportunité'
    df = drop_double_opportunities(df)

    # Filter columns
    keep_cols = [
        "N° Opportunité",
        "Montant",
        "Origine de l'opportunité",
        "Opération",
        "Domaine",
        "Position",
        "Type du compte",
        "Étape",
        "Date de dernière modification",
        "Activité",
        "Rôle du contact",
        "Resp.",
        "Resp..1",
        "Ville",
        "Date de création",
    ]
    df = df[keep_cols]

    df = create_opportunity_age(df)

    # 'Montant' to float, drop NaN in 'Montant', drop negative values in 'Montant'
    df = process_montant(df)

    # One hot encoding
    binary_one_hot_cols = ["Opération", "Domaine", "Position"]
    df = to_binary_one_hot(df, binary_one_hot_cols)

    one_hot_cols = [
        "Origine de l'opportunité",
        "Type du compte",
        "Activité",
        "Rôle du contact",
    ]
    df = to_one_hot(df, one_hot_cols)

    # Create 'Label'
    df = create_labels(df)

    df = join_staff(df)

    df = join_insee_communes(df)

    # df = join_prixm2(df)

    df = df.drop(["Code INSEE"], axis=1)

    if drop_numero_opportunite:
        df = df.drop(["N° Opportunité"], axis=1)

    return df


def load_train_test_split(
    drop_na: bool = False, test_size: float = 0.2, random_state: int = 42
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    opportunities = load_opportunities()
    if drop_na:
        opportunities = opportunities.dropna()

    labels = opportunities["Label"].astype(bool)

    # sample weight is 1 for opportunities younger than 3 years old and 0.2 for the oldest ones, linearly decreasing
    sample_weight = opportunities["opportunity_age"]
    sample_weight = (sample_weight - 3).clip(lower=0)
    sample_weight = (0.2 - 1) / sample_weight.max() * sample_weight + 1

    # add 1 to the sample weight of the positive examples
    sample_weight = sample_weight + labels.astype(int)

    features = opportunities.drop(
        ["Label", "opportunity_age"],
        axis=1,
    )

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        features,
        labels,
        sample_weight,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    return X_train, y_train, w_train, X_test, y_test, w_test


def load_opportunities_for_chantiers() -> pd.DataFrame:
    df = import_opportunities()

    # Drop double in 'N° Opportunité'
    df = drop_double_opportunities(df)

    keep_cols = [
        "N° Opportunité",
        "Montant",
        "Origine de l'opportunité",
        "Opération",
        "Domaine",
        "Position",
        "Type du compte",
        "Étape",
        "Date de dernière modification",
        "Activité",
        "Rôle du contact",
        "Resp.",
        "Resp..1",
        "Ville",
    ]

    df = df[keep_cols]

    # 'Montant' to float, drop NaN in 'Montant', drop negative values in 'Montant'
    df = process_montant(df)

    # One hot encoding

    binary_one_hot_cols = ["Opération", "Domaine", "Position"]
    df = to_binary_one_hot(df, binary_one_hot_cols)

    one_hot_cols = [
        "Origine de l'opportunité",
        "Type du compte",
        "Activité",
        "Rôle du contact",
    ]
    df = to_one_hot(df, one_hot_cols)

    # Create 'Label'

    df = create_labels(df)

    # Clean 'Resp.' and 'Resp..1'

    df.loc[
        (df["Resp..1"] == "LIL") & (df["Resp."] == "Julien MILLE"), "Resp..1"
    ] = "MIL"

    staff = load_staff(df)

    joint = df.join(staff.set_index("Identifiant"), on="Resp..1")

    joint = joint.drop(["Resp..1", "Resp."], axis=1)

    return joint
