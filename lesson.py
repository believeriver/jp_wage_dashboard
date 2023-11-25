import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import List
import pathlib

import warnings
warnings.simplefilter('ignore')

lib_dir = '/Users/tagawanobuyuki/Desktop/Study/udemy/02-python1/lesson/lesson_data_analyze/pdstreamlit/lib/python3.8/site-packages'
sys.path.append(lib_dir)
import japanize_matplotlib
import plotly.express as px
import streamlit as st
import pydeck as pdk


def get_file_name_list(_folder_path) -> List[str]:
    file_name_list = os.listdir(_folder_path)
    return file_name_list


def fetch_csv_to_df(_folder_path, _file_name) -> pd.DataFrame:
    # print('*** import: ', _file_name, '***')
    df = pd.read_csv(_folder_path + '/' + _file_name, encoding='shift_jis')
    return df


def check_print_data(_df: pd.DataFrame):
    print('===== data size \n', _df.shape)
    print('===== data index \n', _df.index)
    print('===== data colum name \n', _df.columns)
    print('===== data type \n', _df.dtypes)
    print(_df.head())
    print(_df.duplicated())
    print(_df.duplicated().any())
    print(_df.isnull().sum())


def replace_str_to_nan(strings: str, _df: pd.DataFrame) -> pd.DataFrame:
    # print([s for s in _df.columns])
    for col in _df.columns:
        _df[col] = _df[col].replace('-', np.nan)
    # print(_df.isnull().sum())

    return _df


def convert_data_type(_df: pd.DataFrame) -> pd.DataFrame:
    # データの変換
    # print(df_pref_category.dtypes)
    _df = _df.astype({
        '所定内給与額（万円）': float,
        '年間賞与その他特別給与額（万円）': float,
        '一人当たり賃金（万円）': float
    })
    # print(_df.dtypes)
    return _df


def sample_select_data(_df: pd.DataFrame) -> None:
    # データの抽出
    print()
    print(_df[(_df['都道府県名'] == '東京都') & (_df['一人当たり賃金（万円）'] > 900)])


def sample_group_by_age(_df: pd.DataFrame) -> None:
    # df_jp_ind_diff = df_jp_ind[df_jp_ind['年齢'] == '年齢計']
    df_jp_ind_diff = _df[_df['年齢'] == '年齢計']
    df_jp_ind_diff['一人当たり賃金の差分（万円）'] = df_jp_ind_diff['一人当たり賃金（万円）'].diff()
    # print(df_jp_ind_diff)

    # print(df_jp_ind.groupby('年齢')['一人当たり賃金（万円）'].mean())
    print(_df.groupby('年齢')[['所定内給与額（万円）', '年間賞与その他特別給与額（万円）', '一人当たり賃金（万円）']].mean())


def sample_group_by_category(_df: pd.DataFrame) -> None:
    # print(_df.head())
    df_temp = _df[_df['年齢'] == '年齢計']
    # print(df_temp)
    print(df_temp.groupby('産業大分類名')[['所定内給与額（万円）', '年間賞与その他特別給与額（万円）', '一人当たり賃金（万円）']].mean())


def sample_group_by_pref(_df: pd.DataFrame) -> None:
    # print(_df.head())
    df_temp = _df[_df['年齢'] == '年齢計']
    # print(df_temp)
    # print(df_temp.groupby('都道府県名')[['所定内給与額（万円）', '年間賞与その他特別給与額（万円）', '一人当たり賃金（万円）']].mean())
    print(df_temp.loc[df_temp['一人当たり賃金（万円）'].idxmax()])


def sample_group_by_pref_and_category(_df: pd.DataFrame) -> None:
    # print(_df.head())
    df_temp = _df[_df['年齢'] == '年齢計']
    df_temp_group = df_temp.groupby(['都道府県名', '産業大分類名'])[['所定内給与額（万円）', '年間賞与その他特別給与額（万円）', '一人当たり賃金（万円）']].mean()
    print(df_temp_group.loc['三重県':'京都府'])


def sample_plot_wage_average(_df: pd.DataFrame) -> None:
    df_ts_mean = _df[_df['年齢'] == '年齢計']
    df_ts_mean = df_ts_mean.set_index('集計年')
    # print(df_ts_mean)
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    # ax = plt.axes()
    ax1.set_xlabel('年')
    ax1.set_ylabel('一人当たり賃金（万円）')
    ax1.plot(df_ts_mean['一人当たり賃金（万円）'])

    ax2.set_xlabel('年')
    ax2.set_ylabel('年間賞与その他特別給与額（万円）')
    ax2.plot(df_ts_mean['年間賞与その他特別給与額（万円）'])

    ax3.set_xlabel('年')
    ax3.set_ylabel('所定内給与額（万円）')
    ax3.set_ylim(10, 40)
    ax3.plot(df_ts_mean['所定内給与額（万円）'])

    plt.show()


def sample_boxplot(_df: pd.DataFrame, target: str) -> None:
    print(_df.head())
    target_list = _df[target].unique()
    # print(target_list)
    wage_list = []
    for item in target_list:
        # print(age)
        wage_temp = _df[_df[target] == item]['一人当たり賃金（万円）'].values.tolist()
        wage_list.append(wage_temp)

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes()
    ax.set_title('年齢階級ごとの一人当たり賃金')
    ax.set_xticklabels(target_list, rotation=90)
    # ax.boxplot(wage_list)
    ax.boxplot(wage_list, showfliers=False)
    plt.show()


def main():
    folder_name = 'csv_data'
    folder_path = str(pathlib.Path.cwd()) + '/' + folder_name
    file_name_list = get_file_name_list(folder_path)

    # import csv data
    df_jp_ind = fetch_csv_to_df(folder_path, file_name_list[0])
    df_jp_category = fetch_csv_to_df(folder_path, file_name_list[1])
    df_pref_ind = fetch_csv_to_df(folder_path, file_name_list[2])
    df_pref_category = fetch_csv_to_df(folder_path, file_name_list[3])

    # 欠損値の処理
    df_pref_category = replace_str_to_nan('-', df_pref_category)
    # 欠損値の削除
    df_pref_category.dropna(subset=['所定内給与額（万円）'], axis=0, inplace=True)
    df_pref_category = convert_data_type(df_pref_category)

    # grouping data
    # sample_group_by_age(df_jp_ind)
    # sample_group_by_category(df_jp_category)
    # sample_group_by_pref(df_pref_ind)
    # sample_group_by_pref_and_category(df_pref_category)

    # plot graph
    # sample_plot_wage_average(df_jp_ind)
    # sample_boxplot(df_pref_category, '年齢')
    sample_boxplot(df_pref_category, '産業大分類名')

    del df_jp_ind, df_pref_ind, df_jp_category, df_pref_category


def main_streamlit():
    def normalize(_df: pd.DataFrame) -> pd.DataFrame:
        return (_df - _df.min()) / (_df.max() - _df.min())

    folder_name = 'csv_data'
    folder_path = str(pathlib.Path.cwd()) + '/' + folder_name
    file_name_list = get_file_name_list(folder_path)

    df_jp_ind = fetch_csv_to_df(folder_path, file_name_list[0])
    df_jp_category = fetch_csv_to_df(folder_path, file_name_list[1])
    df_pref_ind = fetch_csv_to_df(folder_path, file_name_list[2])

    st.title('日本の賃金データダッシュボード')

    """
       マップグラフ
    """
    st.header('■2019年：一人当たり平均賃金のヒートマップ')
    jp_lat_lon = pd.read_csv('./pref_lat_lon.csv')
    jp_lat_lon = jp_lat_lon.rename(columns={'pref_name': '都道府県名'})
    # jp_lat_lon

    df_pref_map = df_pref_ind[
        (df_pref_ind['年齢'] == '年齢計') & (df_pref_ind['集計年'] == 2019)]
    df_pref_map = pd.merge(df_pref_map, jp_lat_lon, on='都道府県名')
    df_pref_map['一人当たり賃金（相対値）'] = normalize(df_pref_map['一人当たり賃金（万円）'])

    view = pdk.ViewState(
        longitude=139.691648,
        latitude=35.68185,
        zoom=4,
        pitch=40.5,
    )

    layer = pdk.Layer(
        "HeatmapLayer",
        data=df_pref_map,
        opacity=0.4,
        get_position=["lon", "lat"],
        threshold=0.3,
        get_weight = '一人当たり賃金（相対値）'
    )

    layer_map = pdk.Deck(
        layers=layer,
        initial_view_state=view,
    )

    st.pydeck_chart(layer_map)

    show_df = st.checkbox('Show DataFrame')
    if show_df is True:
        st.write(df_pref_map)

    """
    line chart
    """
    st.header('■集計年別の一人当たり賃金（万円）の推移')
    df_ts_mean = df_jp_ind[(df_jp_ind['年齢'] == '年齢計')]
    df_ts_mean = df_ts_mean.rename(
        columns={'一人当たり賃金（万円）': '全国_一人あたり賃金（万円）'})

    df_pref_mean = df_pref_ind[(df_pref_ind['年齢'] == '年齢計')]
    pref_list = df_pref_mean['都道府県名'].unique()
    option_pref = st.selectbox('都道府県', pref_list)
    df_pref_mean = df_pref_mean[df_pref_mean['都道府県名'] == option_pref]
    # df_pref_mean

    df_mean_line = pd.merge(df_ts_mean, df_pref_mean, on='集計年')
    df_mean_line = df_mean_line[['集計年', '全国_一人あたり賃金（万円）', '一人当たり賃金（万円）']]
    df_mean_line = df_mean_line.set_index('集計年')
    st.line_chart(df_mean_line)

    """
    バブルグラフ
    """
    st.header('■年齢階級別の全国一人あたり平均賃金（万円）')

    df_mean_bubble = df_jp_ind[df_jp_ind['年齢'] != '年齢計']

    fig = px.scatter(df_mean_bubble,
                     x="一人当たり賃金（万円）",
                     y="年間賞与その他特別給与額（万円）",
                     range_x=[150, 700],
                     range_y=[0, 150],
                     size="所定内給与額（万円）",
                     size_max=38,
                     color="年齢",
                     animation_frame="集計年",
                     animation_group="年齢")
    st.plotly_chart(fig)

    """
        バブルグラフ
        """
    st.header('■産業別の賃金推移')
    year_list = df_jp_category['集計年'].unique()
    option_year = st.selectbox('集計年', year_list)
    wage_list = ['一人当たり賃金（万円）', '所定内給与額（万円）', '年間賞与その他特別給与額（万円）']
    option_wage = st.selectbox('賃金の種類', wage_list)

    df_mean_categ = df_jp_category[(df_jp_category['集計年'] == option_year)]
    max_x = df_mean_categ[option_wage].max() + 50

    fig = px.bar(df_mean_categ,
                 x=option_wage,
                 y='産業大分類名',
                 color='産業大分類名',
                 animation_frame='年齢',
                 range_x=[0, max_x],
                 orientation='h',
                 width=800,
                 height=500)

    st.plotly_chart(fig)

    st.text('出典：RESAS（地域経済分析システム）')
    st.text('本結果はRESAS（地域経済分析システム）を加工して作成')


if __name__ == '__main__':
    # main()
    main_streamlit()