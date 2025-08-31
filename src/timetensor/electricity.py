import datetime
import torch
import pandas as pd

def fetch_txt(path, years=None, hourly=True):
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

    return consumptions, datetimes


def fetch_csv(path, data_name=None, years=None, drop=True, context_cols=None):
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
    datetimes = list(df.index)
    return df, None, datetimes
 


def build_dataset(data_path, raw_format="csv", output_format="torch", years=None, hourly=False, drop=True):
    """fetches correct ECL dataset"""
    if raw_format == "txt":
        values_pt, datetimes = fetch_txt(data_path, years, hourly)
    elif raw_format == "csv":
        #load csv
        values_df, _, datetimes = fetch_csv(data_path, years, drop=drop)
        #tensors
        values_pt = values_df.values
        values_pt = torch.tensor(values_pt, dtype=torch.float32).transpose(1,0).unsqueeze(1) #(individuals, 1, dates)
    else:
        raise ValueError("Format of raw dataset not recognized")
    
    #save
    torch.save(values_pt, data_path + "values.pt")
    torch.save(datetimes, data_path+ "datetimes.pt")