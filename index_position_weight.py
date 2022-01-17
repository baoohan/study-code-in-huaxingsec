# -*- coding: gbk -*-
# Author: HanBao

import os.path
import sys
import optparse
import numpy as np
import pandas as pd
import xarray as xr
from qtdata.panel_func import panel_resize
from qtdata.qtcalendar import get_trading_day_range, get_next_date
from qtdata.read_ics_data import read_ics_data
import qtdata.qtlog as qtlog
from aqtts.fn import Mult
import datetime
from tabulate import tabulate
from qtdata.qtpath import Path
import traceback


def get_options():
    usage = 'usage: %s [-sYYYYMMDD] [-eYYYYMMDD] [-a] [-i]' % (sys.argv[0])
    today = datetime.datetime.now().strftime('%Y%m%d')
    yestday = get_next_date(today, -1)
    path = Path()

    parser = optparse.OptionParser(usage)
    parser.add_option('-s', action='store', dest='start_date', type='string', default=yestday,
                      help='the start date,in format "YYYYMMDD", default yesterday.')
    parser.add_option('-e', action='store', dest='end_date', type='string', default=yestday,
                      help='the end date,in format "YYYYMMDD", default yesterday.')
    parser.add_option('-a', action='store', dest='folder_path', type='string',
                      default=path.public_data_path + 'qtprod/strategy/cap_total_raw',
                      help='strategy folder, default qtprod/strategy/cap_total_raw')
    parser.add_option('-i', action='store', dest='indexSet', type='string',
                      default='CSI300,SZ50,ZZ500,ZZ800,ZZ1000,ZZ1800', help='target univ')
    parser.add_option('-t', action='store', dest='targetStrat', type='string', default=None,
                      help='target strategy, default None (means run for all).')
    options, args = parser.parse_args()

    return options, args


def get_avg_position_of_idx(file_path: str, s: str, e: str,
                            idx_list: list = ('CSI300', 'SZ50', 'ZZ500', 'ZZ800', 'ZZ1000', 'ZZ1800'),
                            ignore_empty=True):
    '''

    :param file_path: 持仓文件路径
    :param s: 开始时间
    :param e: 结束时间
    :param idx_list: 指数列表
    :return: 期间指数平均持仓
    '''

    # get tradeday
    tradeday = pd.DataFrame(get_trading_day_range(s, e), columns=['date'])

    # Initialize idx holding weight
    idx_capital_avgweight = pd.DataFrame(columns=['SH', 'SZ'] + idx_list)
    idx_capital_avgweight.loc[0] = np.zeros(idx_capital_avgweight.shape[1])

    # read stock hold data
    try:
        stock_hold_data = pd.read_csv(file_path, encoding='gbk',
                                      usecols=['ticker', 'date', 'capital', 'exchange_ticker'], sep=',', dtype='str',
                                      comment='#', skipfooter=1, engine='python')
    except:
        try:
            stock_hold_data = pd.read_csv(file_path, encoding='utf-8',
                                          usecols=['ticker', 'date', 'capital', 'exchange_ticker'], sep=',',
                                          dtype='str', comment='#', skipfooter=1, engine='python')
        except UnicodeDecodeError:
            qtlog.error(f'Both gbk and utf-8 cannot decode such cap total raw: {file_path}')
            return

    stock_hold_data = stock_hold_data[(stock_hold_data['date'] >= s) & (stock_hold_data['date'] <= e)]
    if stock_hold_data.shape[0] == 0:
        qtlog.warning(f'Empty cap total raw found in {file_path}, fill with nan.')
        idx_capital_avgweight.loc[0] = np.nan
        return None if ignore_empty else idx_capital_avgweight
    stock_hold = stock_hold_data[stock_hold_data['exchange_ticker'].str.isdigit()]

    # step 1: prepare index universe
    univ_tot = read_ics_data(tradeday['date'].values, dtype='univ', vers='idx')

    # step 2: change stock_hold dataframe into KDTV xarray format
    stock_hold_data_df = stock_hold.set_index(['ticker', 'date'])['capital'].map(float).unstack().fillna(0)
    stock_hold_kdtv = xr.DataArray(stock_hold_data_df.values[:, :, np.newaxis, np.newaxis],
                                   coords={'K': list(stock_hold_data_df.index), 'D': list(stock_hold_data_df.columns),
                                           'T': ["15:00:00.000"], 'V': ['capital']}, dims=['K', 'D', 'T', 'V'])

    # step 3: calculate everyday index holding weight
    stock_hold_total = stock_hold_kdtv.sum(dim='K')
    for idx in idx_list:
        univ_tot_idx = univ_tot.loc[:, :, :, [idx]]
        univ_tot_idx = panel_resize(univ_tot_idx, stock_hold_kdtv, byK=True, byD=True)
        idx_hold_kdtv = Mult(univ_tot_idx, stock_hold_kdtv)
        idx_hold_kdtv_sum = idx_hold_kdtv.sum(dim='K')
        idx_capital_avgweight[idx] = np.nanmean(idx_hold_kdtv_sum.values[:, 0, 0] / stock_hold_total.values[:, 0, 0])

    # step 4: weight in ShangHai/ShenZhen stock exchange
    stock_hold.insert(stock_hold.shape[1], 'stock_exchange', stock_hold['ticker'].str[-2:])
    stock_hold_copy = stock_hold.copy()
    stock_hold_copy['capital'] = stock_hold['capital'].apply(pd.to_numeric)

    total_capital = stock_hold_copy.groupby('date').sum()['capital']

    SH_position = stock_hold_copy[stock_hold_copy['stock_exchange'] == 'SH'].groupby('date').sum()['capital']
    idx_capital_avgweight['SH'] = np.nanmean(SH_position.div(total_capital, axis=0))
    SZ_position = stock_hold_copy[stock_hold_copy['stock_exchange'] == 'SZ'].groupby('date').sum()['capital']
    idx_capital_avgweight['SZ'] = np.nanmean(SZ_position.div(total_capital, axis=0))

    return idx_capital_avgweight


if __name__ == '__main__':
    scriptname = os.path.basename(__file__).split('.')[0]
    qtlog.get_logger(scriptname, True, True)

    options, _args = get_options()
    start_date = options.start_date
    end_date = options.end_date
    folder_path = options.folder_path
    idxlist = options.indexSet
    idx_list = idxlist.split(',')
    idx_position = pd.DataFrame(columns=['SH', 'SZ'] + idx_list)
    target_strat = options.targetStrat
    if target_strat is not None:
        target_strat = target_strat.split(',')

    try:
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if file_name.startswith('AF'):
                    continue
                file_path = root + '/' + file_name
                strategy, file_type = file_name.split('.')
                if file_type != 'txt':
                    continue
                if target_strat is not None and strategy not in target_strat:
                    continue
                tmp_idx_position = get_avg_position_of_idx(file_path, start_date, end_date, idx_list)
                if tmp_idx_position is None:
                    continue
                tmp_idx_position.index = [strategy]
                idx_position = idx_position.append(tmp_idx_position)

        beauty_idx_position = tabulate(idx_position, ['SH', 'SZ'] + idx_list, tablefmt="ortbgl", floatfmt=".1%",
                                       numalign="right")
        qtlog.info(f'\n{beauty_idx_position.__str__()}')
    except Exception:
        qtlog.error(traceback.format_exc())
        sys.exit(-1)
