from MODET.data import preprocessing

ds = preprocessing.Dataset("./export-2020-10-30T16_41_24.239Z.csv")

print(ds.labels[0]["objects"])

