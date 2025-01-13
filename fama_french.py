
import pandas_datareader.data as web


ff_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start='2010')[0].drop('RF', axis=1)


def main():
    print(ff_data)


if __name__ == '__main__':
    main()