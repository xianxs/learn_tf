import os
import pandas as pd

from qi.tool.sql.rdl_sql_loader import sql_factor_loader
import tensorflow as tf

def get_data(
        symbol_: str = '600000.SH',
        from_: str = '20120101',
        to_: str = '20190601',
        resave_: bool = False
) -> pd.DataFrame:
    file = f'{symbol_}.csv'
    if os.path.exists(file) and not resave_:
        df = pd.read_csv(file)
    else:
        col_dict = {
            'prod.prices.close': 'close',
            'prod.prices.open': 'open',
            'prod.prices.hi': 'high',
            'prod.prices.lo': 'low',
            'prod.prices.adjclose': 'adjclose',
            'prod.prices.adjopen': 'adjopen',
            'prod.prices.adjhigh': 'adjhigh',
            'prod.prices.adjlow': 'adjlow',
            'prod.prices.vwap': 'vwap',
            'prod.prices.adjvwap': 'adjvwap',
            'prod.prices.factor': 'adjfactor',
            'prod.prices.volume': 'volume',
        }

        df = sql_factor_loader.get_data_symbol_date(
            fields_=list(col_dict.keys()),
            from_=from_, to_=to_, symbols_=[symbol_], rename_=True
        )

        df.sort_values(by='date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename(columns=col_dict, inplace=True)
        df.to_csv(file, index=False)

    df = pd.read_csv(file)
    return df




def crash_proof():
    """
    in case of GPU CUDA crashing
    """
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        # device_count = {'GPU': 1}
    )
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)


if __name__ == '__main__':
    get_data(resave_=True)
