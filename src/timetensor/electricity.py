import datetime
import torch
import pandas as pd

def fetch_txt_data(path, years=None, hourly=True, return_df=False):
    """returns electricity.txt dataset as consumptions tensor and datetimes list"""
    consumptions = [] #conso des individus
    datetimes = [] #informations sur la date

    with open(path + "electricity.txt", 'r') as file:
        next(file)
        for line in file:
            parts = line.strip().split(';')
            dt = datetime.datetime.strptime(parts[0].strip('"'), "%Y-%m-%d %H:%M:%S")
            if years is None or dt.year in years: #filtre sur l'année
                datetimes.append(dt)
                consumptions.append([float(parts[k].replace(",", ".")) for k in range(1,len(parts))])

    consumptions = torch.tensor(consumptions, dtype=torch.float32).transpose(1,0).unsqueeze(1) #(N_individuals, 1, N_dates)

    if hourly: #pas de temps 15 min => pas horaire
        N_individuals, dim, N_dates = consumptions.shape
        consumptions = consumptions.view(N_individuals, dim, N_dates//4, 4).sum(dim=3)
        datetimes = datetimes[::4]

    if return_df:
        return pd.DataFrame(consumptions.squeeze(0).transpose(1,0), index=datetimes)
    return consumptions, datetimes


def fetch_csv_data(path, years=None, return_df=False, drop=True):
    """returns electricity.csv dataset as consumptions tensor and datetimes list"""
    #df = pd.read_csv(path + "electricity.csv")
    #datetimes = [datetime.datetime.strptime(date.strip('"'), "%Y-%m-%d %H:%M:%S") for date in df.date]
    df = pd.read_csv(path+'electricity.csv', index_col=0, parse_dates=True)
    if drop:
        df = df.drop(columns=["57", "106", "127", "182", "298"]) #big missing values
    df = df.rename(columns={"OT":"320"})
    if years is not None:
        df = df[df.index.year.isin(years)]
    df.columns = range(df.shape[1])
    if return_df:
        return df
    else:
        consumptions = df.values #df.drop(columns=["date"]).values
        datetimes = list(df.index)
        consumptions = torch.tensor(consumptions, dtype=torch.float32).transpose(1,0).unsqueeze(1) #(N_individuals, 1, N_dates)
        return consumptions, datetimes


def fetch_data(path, raw_format="csv", output_format="torch", years=None, hourly=False, drop=True):
    """fetches correct dataset"""
    if output_format == "pandas":
        if raw_format == "txt":
            return fetch_txt_data(path, years, hourly, return_df=True)
        elif raw_format == "csv":
            return fetch_csv_data(path, years, return_df=True, drop=drop)
        else:
            raise ValueError("Format of raw dataset not recognized")
    else:
        if raw_format == "txt":
            consumptions, datetimes = fetch_txt_data(path, years, hourly, return_df=False)
        elif raw_format == "csv":
            consumptions, datetimes = fetch_csv_data(path, years, return_df=False, drop=drop)
        else:
            raise ValueError("Format of raw dataset not recognized")
        return consumptions, None, datetimes