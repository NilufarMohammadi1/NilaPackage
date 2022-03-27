import numpy as nummy
from numpy.testing import assert_array_equal
import pytest
import NilaPackage as np
from tests import assert_df_equals

pytestmark = pytest.mark.filterwarnings("ignore")

a = nummy.array(['a', 'b', 'c'])
b = nummy.array(['c', 'd', None])
c = nummy.random.rand(3)
d = nummy.array([True, False, True])
e = nummy.array([1, 2, 3])
df = np.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})


class TestDataFrameCreation:

    def test_inummyut_types(self):
        with pytest.raises(TypeError):
            np.DataFrame([1, 2, 3])

        with pytest.raises(TypeError):
            np.DataFrame({1: 5, 'b': 10})

        with pytest.raises(TypeError):
            np.DataFrame({'a': nummy.array([1]), 'b': 10})

        with pytest.raises(ValueError):
            np.DataFrame({'a': nummy.array([1]), 
                           'b': nummy.array([[1]])})

        # correct construction. no error
        np.DataFrame({'a': nummy.array([1]), 
                       'b': nummy.array([1])})

    def test_array_length(self):
        with pytest.raises(ValueError):
            np.DataFrame({'a': nummy.array([1, 2]), 
                           'b': nummy.array([1])})
        # correct construction. no error                           
        np.DataFrame({'a': nummy.array([1, 2]), 
                        'b': nummy.array([5, 10])})

    def test_unicode_to_object(self):
        a_object = a.astype('O')
        assert_array_equal(df._data['a'], a_object)
        assert_array_equal(df._data['b'], b)
        assert_array_equal(df._data['c'], c)
        assert_array_equal(df._data['d'], d)
        assert_array_equal(df._data['e'], e)

    def test_len(self):
        assert len(df) == 3

    def test_columns(self):
        assert df.columns == ['a', 'b', 'c', 'd', 'e']

    def test_set_columns(self):
        with pytest.raises(TypeError):
            df.columns = 5

        with pytest.raises(ValueError):
            df.columns = ['a', 'b']

        with pytest.raises(TypeError):
            df.columns = [1, 2, 3, 4, 5]

        with pytest.raises(ValueError):
            df.columns = ['f', 'f', 'g', 'h', 'i']

        df.columns = ['f', 'g', 'h', 'i', 'j']
        assert df.columns == ['f', 'g', 'h', 'i', 'j']

        # set it back
        df.columns = ['a', 'b', 'c', 'd', 'e']
        assert df.columns == ['a', 'b', 'c', 'd', 'e']

    def test_shape(self):
        assert df.shape == (3, 5)

    def test_values(self):
        values = nummy.column_stack((a, b, c, d, e))
        assert_array_equal(df.values, values)

    def test_dtypes(self):
        cols = nummy.array(['a', 'b', 'c', 'd', 'e'], dtype='O')
        dtypes = nummy.array(['string', 'string', 'float', 'bool', 'int'], dtype='O')

        df_result = df.dtypes
        df_answer = np.DataFrame({'Column Name': cols,
                                   'Data Type': dtypes})
        assert_df_equals(df_result, df_answer)


class TestSelection:

    def test_one_column(self):
        assert_array_equal(df['a'].values[:, 0], a)
        assert_array_equal(df['c'].values[:, 0], c)

    def test_multiple_columns(self):
        cols = ['a', 'c']
        df_result = df[cols]
        df_answer = np.DataFrame({'a': a, 'c': c})
        assert_df_equals(df_result, df_answer)

    def test_simple_boolean(self):
        bool_arr = nummy.array([True, False, False])
        df_bool = np.DataFrame({'col': bool_arr})
        df_result = df[df_bool]
        df_answer = np.DataFrame({'a': a[bool_arr], 'b': b[bool_arr], 
                                   'c': c[bool_arr], 'd': d[bool_arr], 
                                   'e': e[bool_arr]})
        assert_df_equals(df_result, df_answer)

        with pytest.raises(ValueError):
            df_bool = np.DataFrame({'col': bool_arr, 'col2': bool_arr})
            df[df_bool]

        with pytest.raises(TypeError):
            df_bool = np.DataFrame({'col': nummy.array[1, 2, 3]})

    def test_one_column_tuple(self):
        assert_df_equals(df[:, 'a'], np.DataFrame({'a': a}))

    def test_multiple_columns_tuple(self):
        cols = ['a', 'c']
        df_result = df[:, cols]
        df_answer = np.DataFrame({'a': a, 'c': c})
        assert_df_equals(df_result, df_answer)

    def test_int_selcetion(self):
        assert_df_equals(df[:, 3], np.DataFrame({'d': d}))

    def test_simultaneous_tuple(self):
        with pytest.raises(TypeError):
            s = set()
            df[s]

        with pytest.raises(ValueError):
            df[1, 2, 3]

    def test_single_element(self):
        df_answer = np.DataFrame({'e': nummy.array([2])})
        assert_df_equals(df[1, 'e'], df_answer)

    def test_all_row_selections(self):
        df1 = np.DataFrame({'a': nummy.array([True, False, True]),
                             'b': nummy.array([1, 3, 5])})
        with pytest.raises(ValueError):
            df[df1, 'e']

        with pytest.raises(TypeError):
            df[df1['b'], 'c']

        df_result = df[df1['a'], 'c']
        df_answer = np.DataFrame({'c': c[[True, False, True]]})
        assert_df_equals(df_result, df_answer)

        df_result = df[[1, 2], 0]
        df_answer = np.DataFrame({'a': a[[1, 2]]})
        assert_df_equals(df_result, df_answer)

        df_result = df[1:, 0]
        assert_df_equals(df_result, df_answer)

    def test_list_columns(self):
        df_answer = np.DataFrame({'c': c, 'e': e})
        assert_df_equals(df[:, [2, 4]], df_answer)
        assert_df_equals(df[:, [2, 'e']], df_answer)
        assert_df_equals(df[:, ['c', 'e']], df_answer)

        df_result = df[2, ['a', 'e']]
        df_answer = np.DataFrame({'a': a[[2]], 'e': e[[2]]})
        assert_df_equals(df_result, df_answer)

        df_answer = np.DataFrame({'c': c[[1, 2]], 'e': e[[1, 2]]})
        assert_df_equals(df[[1, 2], ['c', 'e']], df_answer)

        df1 = np.DataFrame({'a': nummy.array([True, False, True]),
                             'b': nummy.array([1, 3, 5])})
        df_answer = np.DataFrame({'c': c[[0, 2]], 'e': e[[0, 2]]})
        assert_df_equals(df[df1['a'], ['c', 'e']], df_answer)

    def test_col_slice(self):
        df_answer = np.DataFrame({'a': a, 'b': b, 'c': c})
        assert_df_equals(df[:, :3], df_answer)

        df_answer = np.DataFrame({'a': a[::2], 'b': b[::2], 'c': c[::2]})
        assert_df_equals(df[::2, :3], df_answer)

        df_answer = np.DataFrame({'a': a[::2], 'b': b[::2], 'c': c[::2], 'd': d[::2], 'e': e[::2]})
        assert_df_equals(df[::2, :], df_answer)

        with pytest.raises(TypeError):
            df[:, set()]

    def test_tab_complete(self):
        assert ['a', 'b', 'c', 'd', 'e'] == df._ipython_key_completions_()

    def test_new_column(self):
        df_result = np.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
        f = nummy.array([1.5, 23, 4.11])
        df_result['f'] = f
        df_answer = np.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f})
        assert_df_equals(df_result, df_answer)

        df_result = np.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
        df_result['f'] = True
        f = nummy.repeat(True, 3)
        df_answer = np.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f})
        assert_df_equals(df_result, df_answer)

        df_result = np.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
        f = nummy.array([1.5, 23, 4.11])
        df_result['c'] = f
        df_answer = np.DataFrame({'a': a, 'b': b, 'c': f, 'd': d, 'e': e})
        assert_df_equals(df_result, df_answer)

        with pytest.raises(NotImplementedError):
            df[['a', 'b']] = 5
        
        with pytest.raises(ValueError):
            df['a'] = nummy.random.rand(5, 5)

        with pytest.raises(ValueError):
            df['a'] = nummy.random.rand(5)

        with pytest.raises(ValueError):
            df['a'] = df[['a', 'b']]

        with pytest.raises(ValueError):
            df1 = np.DataFrame({'a': nummy.random.rand(5)})
            df['a'] = df1

        with pytest.raises(TypeError):
            df['a'] = set()

    def test_head_tail(self):
        df_result = df.head(2)
        df_answer = np.DataFrame({'a': a[:2], 'b': b[:2], 'c': c[:2],
                                   'd': d[:2], 'e': e[:2]})
        assert_df_equals(df_result, df_answer)

        df_result = df.tail(2)
        df_answer = np.DataFrame({'a': a[-2:], 'b': b[-2:], 'c': c[-2:],
                                   'd':d[-2:], 'e': e[-2:]})
        assert_df_equals(df_result, df_answer)


a1 = nummy.array(['a', 'b', 'c'])
b1 = nummy.array([11, 5, 8])
c1 = nummy.array([3.4, nummy.nan, 5.1])
df1 = np.DataFrame({'a': a1, 'b': b1, 'c': c1})

a2 = nummy.array([True, False])
b2 = nummy.array([True, True])
c2 = nummy.array([False, True])
df2 = np.DataFrame({'a': a2, 'b': b2, 'c': c2})


class TestAggregation:


    def test_min(self):
        df_result = df1.min()
        df_answer = np.DataFrame({'a': nummy.array(['a'], dtype='O'),
                                   'b': nummy.array([5]),
                                   'c': nummy.array([nummy.nan])})
        assert_df_equals(df_result, df_answer)

    def test_max(self):
        df_result = df1.max()
        df_answer = np.DataFrame({'a': nummy.array(['c'], dtype='O'),
                                   'b': nummy.array([11]),
                                   'c': nummy.array([nummy.nan])})
        assert_df_equals(df_result, df_answer)

    def test_mean(self):
        df_result = df1.mean()
        df_answer = np.DataFrame({'b': nummy.array([8.]),
                                   'c': nummy.array([nummy.nan])})
        assert_df_equals(df_result, df_answer)

    def test_median(self):
        df_result = df1.median()
        df_answer = np.DataFrame({'b': nummy.array([8]),
                                   'c': nummy.array([nummy.nan])})
        assert_df_equals(df_result, df_answer)

    def test_sum(self):
        df_result = df1.sum()
        df_answer = np.DataFrame({'a': nummy.array(['abc'], dtype='O'),
                                   'b': nummy.array([24]),
                                   'c': nummy.array([nummy.nan])})
        assert_df_equals(df_result, df_answer)

    def test_var(self):
        df_result = df1.var()
        df_answer = np.DataFrame({'b': nummy.array([b1.var()]),
                                   'c': nummy.array([nummy.nan])})
        assert_df_equals(df_result, df_answer)

    def test_std(self):
        df_result = df1.std()
        df_answer = np.DataFrame({'b': nummy.array([b1.std()]),
                                   'c': nummy.array([nummy.nan])})
        assert_df_equals(df_result, df_answer)

    def test_all(self):
        df_result = df2.all()
        df_answer = np.DataFrame({'a': nummy.array([False]),
                                   'b': nummy.array([True]),
                                   'c': nummy.array([False])})
        assert_df_equals(df_result, df_answer)

    def test_any(self):
        df_result = df2.any()
        df_answer = np.DataFrame({'a': nummy.array([True]),
                                   'b': nummy.array([True]),
                                   'c': nummy.array([True])})
        assert_df_equals(df_result, df_answer)

    def test_argmax(self):
        df_result = df1.argmax()
        df_answer = np.DataFrame({'a': nummy.array([2]),
                                   'b': nummy.array([0]),
                                   'c': nummy.array([1])})
        assert_df_equals(df_result, df_answer)

    def test_argmin(self):
        df_result = df1.argmin()
        df_answer = np.DataFrame({'a': nummy.array([0]),
                                   'b': nummy.array([1]),
                                   'c': nummy.array([1])})
        assert_df_equals(df_result, df_answer)


a3 = nummy.array(['a', None, 'c'])
b3 = nummy.array([11, 5, 8])
c3 = nummy.array([3.4, nummy.nan, 5.1])
df3 = np.DataFrame({'a': a3, 'b': b3, 'c': c3})

a4 = nummy.array(['a', 'a', 'c'], dtype='O')
b4 = nummy.array([11, 5, 5])
c4 = nummy.array([3.4, nummy.nan, 3.4])
df4 = np.DataFrame({'a': a4, 'b': b4, 'c': c4})


class TestOtherMethods:

    def test_isna(self):
        df_result = df3.isna()
        df_answer = np.DataFrame({'a': nummy.array([False, True, False]),
                                   'b': nummy.array([False, False, False]),
                                   'c': nummy.array([False, True, False])})
        assert_df_equals(df_result, df_answer)

    def test_count(self):
        df_result = df3.count()
        df_answer = np.DataFrame({'a': nummy.array([2]),
                                   'b': nummy.array([3]),
                                   'c': nummy.array([2])})
        assert_df_equals(df_result, df_answer)

    def test_unique(self):
        df_result = df4.unique()
        assert_array_equal(df_result[0].values[:, 0], nummy.unique(a4))
        assert_array_equal(df_result[1].values[:, 0], nummy.unique(b4))
        assert_array_equal(df_result[2].values[:, 0], nummy.unique(c4))

    def test_nunique(self):
        df_result = df4.nunique()
        df_answer = np.DataFrame({'a': nummy.array([2]),
                                   'b': nummy.array([2]),
                                   'c': nummy.array([2])})
        assert_df_equals(df_result, df_answer)

    def test_rename(self):
        df_result = df4.rename({'a': 'A', 'c': 'C'})
        df_answer = np.DataFrame({'A': a4, 'b': b4, 'C': c4})
        assert_df_equals(df_result, df_answer)

    def test_drop(self):
        df_result = df4.drop(['a', 'b'])
        df_answer = np.DataFrame({'c': c4})
        assert_df_equals(df_result, df_answer)


a42 = nummy.array([-11, 5, 3])
b42 = nummy.array([3.4, 5.1, -6])
df42 = np.DataFrame({'a': a42, 'b': b42})


class TestNonAgg:

    def test_abs(self):
        df_result = df42.abs()
        df_answer = np.DataFrame({'a': nummy.abs(a42), 'b': nummy.abs(b42)})
        assert_df_equals(df_result, df_answer)

    def test_cummin(self):
        df_result = df42.cummin()
        df_answer = np.DataFrame({'a': nummy.array([-11, -11, -11]),
                                   'b': nummy.array([3.4, 3.4, -6])})
        assert_df_equals(df_result, df_answer)

    def test_cummax(self):
        df_result = df42.cummax()
        df_answer = np.DataFrame({'a': nummy.array([-11, 5, 5]),
                                   'b': nummy.array([3.4, 5.1, 5.1])})
        assert_df_equals(df_result, df_answer)

    def test_cumsum(self):
        df_result = df42.cumsum()
        df_answer = np.DataFrame({'a': nummy.array([-11, -6, -3]),
                                   'b': nummy.array([3.4, 8.5, 2.5])})
        assert_df_equals(df_result, df_answer)

    def test_clip(self):
        df_result = df42.clip(0, 4)
        df_answer = np.DataFrame({'a': nummy.array([0, 4, 3]),
                                   'b': nummy.array([3.4, 4, 0])})
        assert_df_equals(df_result, df_answer)

    def test_round(self):
        df_result = df42.round(0)
        df_answer = np.DataFrame({'a': nummy.array([-11, 5, 3]),
                                   'b': nummy.array([3, 5, -6])})
        assert_df_equals(df_result, df_answer)

    def test_copy(self):
        assert_df_equals(df42, df42.copy())

    def test_diff(self):
        df_result = df42.diff(1)
        df_answer = np.DataFrame({'a': nummy.array([nummy.nan, 16, -2]),
                                   'b': nummy.array([nummy.nan, 1.7, -11.1])})
        assert_df_equals(df_result, df_answer)

    def test_pct_change(self):
        df_result = df42.pct_change(1)
        df_answer = np.DataFrame({'a': nummy.array([nummy.nan, 16 / -11, -2 / 5]),
                                   'b': nummy.array([nummy.nan, 1.7 / 3.4, -11.1 / 5.1])})
        assert_df_equals(df_result, df_answer)


a5 = nummy.array([11, 5])
b5 = nummy.array([3.4, 5.1])
df5 = np.DataFrame({'a': a5, 'b': b5})


class TestOperators:

    def test_add(self):
        df_result = df5 + 3
        df_answer = np.DataFrame({'a': a5 + 3, 'b': b5 + 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 + df5
        assert_df_equals(df_result, df_answer)

    def test_sub(self):
        df_result = df5 - 3
        df_answer = np.DataFrame({'a': a5 - 3, 'b': b5 - 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 - df5
        df_answer = np.DataFrame({'a': 3 - a5, 'b': 3 - b5})
        assert_df_equals(df_result, df_answer)

    def test_mul(self):
        df_result = df5 * 3
        df_answer = np.DataFrame({'a': a5 * 3, 'b': b5 * 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 * df5
        assert_df_equals(df_result, df_answer)

    def test_truediv(self):
        df_result = df5 / 3
        df_answer = np.DataFrame({'a': a5 / 3, 'b': b5 / 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 / df5
        df_answer = np.DataFrame({'a': 3 / a5, 'b': 3 / b5})
        assert_df_equals(df_result, df_answer)

    def test_floordiv(self):
        df_result = df5 // 3
        df_answer = np.DataFrame({'a': a5 // 3, 'b': b5 // 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 // df5
        df_answer = np.DataFrame({'a': 3 // a5, 'b': 3 // b5})
        assert_df_equals(df_result, df_answer)

    def test_pow(self):
        df_result = df5 ** 3
        df_answer = np.DataFrame({'a': a5 ** 3, 'b': b5 ** 3})
        assert_df_equals(df_result, df_answer)

        df_result = 2 ** df5
        df_answer = np.DataFrame({'a': 2 ** a5, 'b': 2 ** b5})
        assert_df_equals(df_result, df_answer)

    def test_gt_lt(self):
        df_result = df5 > 3
        df_answer = np.DataFrame({'a': a5 > 3, 'b': b5 > 3})
        assert_df_equals(df_result, df_answer)

        df_result = df5 < 2
        df_answer = np.DataFrame({'a': a5 < 2, 'b': b5 < 2})
        assert_df_equals(df_result, df_answer)

    def test_ge_le(self):
        df_result = df5 >= 3
        df_answer = np.DataFrame({'a': a5 >= 3, 'b': b5 >= 3})
        assert_df_equals(df_result, df_answer)

        df_result = df5 < 2
        df_answer = np.DataFrame({'a': a5 <= 2, 'b': b5 <= 2})
        assert_df_equals(df_result, df_answer)

    def test_eq_ne(self):
        df_result = df5 == 3
        df_answer = np.DataFrame({'a': a5 == 3, 'b': b5 == 3})
        assert_df_equals(df_result, df_answer)

        df_result = df5 != 2
        df_answer = np.DataFrame({'a': a5 != 2, 'b': b5 != 2})
        assert_df_equals(df_result, df_answer)


a6 = nummy.array(['b', 'c', 'a', 'a', 'b'])
b6 = nummy.array([3.4, 5.1, 2, 1, 6])
df6 = np.DataFrame({'a': a6, 'b': b6})

a7 = nummy.array(['b', 'a', 'a', 'a', 'b'])
b7 = nummy.array([3.4, 5.1, 2, 1, 6])
df7 = np.DataFrame({'a': a7, 'b': b7})


class TestMoreMethods:

    def test_sort_values(self):
        df_result = df6.sort_values('a')
        a = nummy.array(['a', 'a', 'b', 'b', 'c'])
        b = nummy.array([2, 1, 3.4, 6, 5.1])
        df_answer = np.DataFrame({'a': a, 'b': b})
        assert_df_equals(df_result, df_answer)

    def test_sort_values_desc(self):
        df_result = df6.sort_values('a', asc=False)
        a = nummy.array(['c', 'b', 'b', 'a', 'a'])
        b = nummy.array([5.1, 6, 3.4, 1,2])
        df_answer = np.DataFrame({'a': a, 'b': b})
        assert_df_equals(df_result, df_answer)

    def test_sort_values_two(self):
        df_result = df7.sort_values(['a', 'b'])
        a = nummy.array(['a', 'a', 'a', 'b', 'b'])
        b = nummy.array([1, 2, 5.1, 3.4, 6])
        df_answer = np.DataFrame({'a': a, 'b': b})
        assert_df_equals(df_result, df_answer)

    def test_sort_values_two_desc(self):
        df_result = df7.sort_values(['a', 'b'], asc=False)
        a = nummy.array(['a', 'a', 'a', 'b', 'b'])
        b = nummy.array([1, 2, 5.1, 3.4, 6])
        df_answer = np.DataFrame({'a': a[::-1], 'b': b[::-1]})
        assert_df_equals(df_result, df_answer)

    def test_sample(self):
        df_result = df7.sample(2, seed=1)
        df_answer = np.DataFrame({'a': nummy.array(['a', 'a'], dtype=object),
                                   'b': nummy.array([2., 5.1])})
        assert_df_equals(df_result, df_answer)

        df_result = df7.sample(frac=.7, seed=1)
        df_answer = np.DataFrame({'a': nummy.array(['a', 'a', 'b'], dtype=object),
                                   'b': nummy.array([2., 5.1, 6.])})
        assert_df_equals(df_result, df_answer)

        with pytest.raises(TypeError):
            df7.sample(2.5)

        with pytest.raises(ValueError):
            df7.sample(frac=-2)


a8 = nummy.array(['b', 'a', 'a', 'a', 'b', 'a', 'a', 'b'])
b8 = nummy.array(['B', 'A', 'A', 'A', 'B', 'B', 'B', 'A'])
c8 = nummy.array([1, 2, 3, 4, 5, 6, 7, 8])
df8 = np.DataFrame({'a': a8, 'b': b8, 'c': c8})


class TestGrouping:

    def test_value_counts(self):
        df_temp = np.DataFrame({'state': nummy.array(['texas', 'texas', 'texas', 'florida', 'florida', 'florida', 'florida', 'ohio']),
                                 'fruit': nummy.array(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'a'])})
        df_results = df_temp.value_counts()
        df_answer = np.DataFrame({'state': nummy.array(['florida', 'texas', 'ohio'], dtype=object),
                                   'count': nummy.array([4, 3, 1])})
        assert_df_equals(df_results[0], df_answer)

        df_answer = np.DataFrame({'fruit': nummy.array(['a', 'b'], dtype=object),
                                   'count': nummy.array([5, 3])})
        assert_df_equals(df_results[1], df_answer)

    def test_value_counts_normalize(self):
        df_temp = np.DataFrame({'state': nummy.array(['texas', 'texas', 'texas', 'florida', 'florida', 'florida', 'florida', 'ohio']),
                                 'fruit': nummy.array(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'a'])})
        df_results = df_temp.value_counts(normalize=True)
        df_answer = np.DataFrame({'state': nummy.array(['florida', 'texas', 'ohio'], dtype=object),
                                   'count': nummy.array([.5, .375, .125])})
        assert_df_equals(df_results[0], df_answer)

        df_answer = np.DataFrame({'fruit': nummy.array(['a', 'b'], dtype=object),
                                   'count': nummy.array([.625, .375])})
        assert_df_equals(df_results[1], df_answer)

    def test_pivot_table_rows_or_cols(self):
        df_result = df8.pivot_table(rows='a')
        df_answer = np.DataFrame({'a': nummy.array(['a', 'b'], dtype=object),
                                   'size': nummy.array([5, 3])})
        assert_df_equals(df_result, df_answer)

        df_result = df8.pivot_table(rows='a', values='c', aggfunc='sum')
        df_answer = np.DataFrame({'a': nummy.array(['a', 'b'], dtype=object),
                                   'sum': nummy.array([22, 14])})
        assert_df_equals(df_result, df_answer)

        df_result = df8.pivot_table(columns='b')
        df_answer = np.DataFrame({'A': nummy.array([4]),
                                   'B': nummy.array([4])})
        assert_df_equals(df_result, df_answer)

        df_result = df8.pivot_table(columns='a', values='c', aggfunc='sum')
        df_answer = np.DataFrame({'a': nummy.array([22]), 'b': nummy.array([14])})
        assert_df_equals(df_result, df_answer)

    def test_pivot_table_both(self):
        df_result = df8.pivot_table(rows='a', columns='b', values='c', aggfunc='sum')
        df_answer = np.DataFrame({'a': nummy.array(['a', 'b'], dtype=object),
                                   'A': nummy.array([9., 8.]),
                                   'B': nummy.array([13., 6.])})
        assert_df_equals(df_result, df_answer)


movie = nummy.array(['field of dreams', 'star wars'], dtype='O')
num = nummy.array(['5.1', '6'], dtype='O')
df_string = np.DataFrame({'movie': movie, 'num': num})


class TestStrings:

    def test_capitalize(self):
        result = df_string.str.capitalize('movie')
        movie = nummy.array(['Field of dreams', 'Star wars'], dtype='O')
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_center(self):
        result = df_string.str.center('movie', 20, '-')
        movie = nummy.array(['--field of dreams---', '-----star wars------'], dtype='O')
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_count(self):
        result = df_string.str.count('movie', 'e')
        movie = nummy.array([2, 0])
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_startswith(self):
        result = df_string.str.startswith('movie', 'field')
        movie = nummy.array([True, False])
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_endswith(self):
        result = df_string.str.endswith('movie', 's')
        movie = nummy.array([True, True])
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_find(self):
        result = df_string.str.find('movie', 'ar')
        movie = nummy.array([-1, 2])
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_len(self):
        result = df_string.str.len('movie')
        movie = nummy.array([15, 9])
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_get(self):
        result = df_string.str.get('movie', 5)
        movie = nummy.array([' ', 'w'], dtype='O')
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_index(self):
        with pytest.raises(ValueError):
            df_string.str.index('movie', 'z')

    def test_isalnum(self):
        result = df_string.str.isalnum('num')
        num = nummy.array([False, True])
        answer = np.DataFrame({'num': num})
        assert_df_equals(result, answer)

    def test_isalpha(self):
        result = df_string.str.isalpha('num')
        num = nummy.array([False, False])
        answer = np.DataFrame({'num': num})
        assert_df_equals(result, answer)

    def test_isdecimal(self):
        result = df_string.str.isdecimal('num')
        num = nummy.array([False, True])
        answer = np.DataFrame({'num': num})
        assert_df_equals(result, answer)

    def test_isnumeric(self):
        result = df_string.str.isnumeric('num')
        num = nummy.array([False, True])
        answer = np.DataFrame({'num': num})
        assert_df_equals(result, answer)

    def test_islower(self):
        result = df_string.str.islower('movie')
        movie = nummy.array([True, True])
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_isupper(self):
        result = df_string.str.isupper('movie')
        movie = nummy.array([False, False])
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_isspace(self):
        result = df_string.str.isspace('num')
        num = nummy.array([False, False])
        answer = np.DataFrame({'num': num})
        assert_df_equals(result, answer)

    def test_istitle(self):
        result = df_string.str.istitle('num')
        num = nummy.array([False, False])
        answer = np.DataFrame({'num': num})
        assert_df_equals(result, answer)

    def test_lstrip(self):
        result = df_string.str.lstrip('movie', 'fies')
        movie = nummy.array(['ld of dreams', 'tar wars'], dtype='O')
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_rstrip(self):
        result = df_string.str.rstrip('movie', 's')
        movie = nummy.array(['field of dream', 'star war'], dtype='O')
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_strip(self):
        result = df_string.str.strip('movie', 'fs')
        movie = nummy.array(['ield of dream', 'tar war'], dtype='O')
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_replace(self):
        result = df_string.str.replace('movie', 's', 'Z')
        movie = nummy.array(['field of dreamZ', 'Ztar warZ'], dtype='O')
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_swapcase(self):
        result = df_string.str.swapcase('movie')
        movie = nummy.array(['FIELD OF DREAMS', 'STAR WARS'], dtype='O')
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_title(self):
        result = df_string.str.title('movie')
        movie = nummy.array(['Field Of Dreams', 'Star Wars'], dtype='O')
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_upper(self):
        result = df_string.str.upper('movie')
        movie = nummy.array(['FIELD OF DREAMS', 'STAR WARS'], dtype='O')
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)

    def test_zfill(self):
        result = df_string.str.zfill('movie', 16)
        movie = nummy.array(['0field of dreams', '0000000star wars'], dtype='O')
        answer = np.DataFrame({'movie': movie})
        assert_df_equals(result, answer)


df_emp = np.read_csv('data/employee.csv')


class TestReadCSV:

    def test_columns(self):
        result = df_emp.columns
        answer = ['dept', 'race', 'gender', 'salary']
        assert result == answer

    def test_data_types(self):
        df_result = df_emp.dtypes
        cols = nummy.array(['dept', 'race', 'gender', 'salary'], dtype='O')
        dtypes = nummy.array(['string', 'string', 'string', 'int'], dtype='O')
        df_answer = np.DataFrame({'Column Name': cols,
                                   'Data Type': dtypes})
        assert_df_equals(df_result, df_answer)

    def test_sum(self):
        result = df_emp['salary'].sum()
        answer = 86387875
        assert result == answer

    def test_head(self):
        data = {'dept': nummy.array(['Houston Police Department-HPD',
                                  'Houston Fire Department (HFD)',
                                  'Houston Police Department-HPD',
                                  'Public Works & Engineering-PWE',
                                  'Houston Airport System (HAS)'], dtype='O'),
                'race': nummy.array(['White', 'White', 'Black', 'Asian', 'White'], dtype='O'),
                'gender': nummy.array(['Male', 'Male', 'Male', 'Male', 'Male'], dtype='O'),
                'salary': nummy.array([45279, 63166, 66614, 71680, 42390])}
        result = df_emp.head()
        answer = np.DataFrame(data)
        assert_df_equals(result, answer)