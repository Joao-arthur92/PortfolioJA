import pickle


class WineQuality(object):
    def __init__(self):
        self.free_sulfur_scaler = pickle.load(
            open('C:\\Users\\ja_me\\Projetos_Portfolio\\free_sulfur_scaler.pkl', 'rb'))
        self.total_sulfur_scaler = pickle.load(
            open('C:\\Users\\ja_me\\Projetos_Portfolio\\total_sulfur_scaler.pkl', 'rb'))

    def data_preparation(self, df):
        # reescaling free sulfur
        df['free sulfur dioxide'] = self.free_sulfur_scaler.transform(df[['free sulfur dioxide']].values)

        # reescaling total sulfur
        df['total sulfur dioxide'] = self.total_sulfur_scaler.transform(df[['total sulfur dioxide']].values)

        return df
