from imblearn.over_sampling import BorderlineSMOTE
import pickle


def generate_data():
    ori_d = pickle.load(open('ori_data.pkl', 'rb'))
    smo_d = []
    smo = BorderlineSMOTE(kind="borderline-2")

    x = [x[:-1] for x in ori_d]
    y = [y[-1] for y in ori_d]
    x_smo, y_smo = smo.fit_resample(x, y)
    
    for i in range(len(y_smo)):
        smo_d.append(x_smo[i])
        smo_d[i].append(y_smo[i])

    result_file = "smo_data.pkl"
    with open(result_file, "wb") as fp:
        pickle.dump(smo_d, fp)


if __name__ == "__main__":
    generate_data()
