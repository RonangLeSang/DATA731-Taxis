
def tauxErreure(lignedf):
    valeurpratique = lignedf["trip_duration"]
    valeurEstimer = "fonction de prediction pour la ligne "
    difference = (valeurpratique - valeurEstimer).dt.total_seconds()
    pourcentage = (difference / valeurpratique.dt.total_seconds()) * 100


def moyenneErreure(dataframe):
    moyenne = 0
    lignes = 0
    for index, row in dataframe.iterrows():
        moyenne += tauxErreure(row)
        lignes += 1
    return moyenne / lignes
