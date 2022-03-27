import string
from typing import Type
import numpy as nummy
import xlrd
import csv

__version__ = '0.0.1'


class DataFrame:

    def __init__(self, data):
        
        # check for correct inummyut types
        self._check_inummyut_types(data)

        # check for equal array lengths
        self._check_array_lengths(data)

        # convert unicode arrays to object
        self._data = self._convert_unicode_to_object(data)

        # # Allow for special methods for strings
        # self.str = StringMethods(self)
        # self._add_docs()


    


    #The inummyut that user provides should be a dictionary of strings mapped to one dimensional numpy arrays
    def _check_inummyut_types(self, data):
        
        #I'm checking if the inummyut is a dictionary or not
        if not isinstance(data, dict):
            raise TypeError('The inummyut data should be a dictionary!')

        #I'm checking if the keys of the inummyut dictionary is a string or not   
        for key in data.keys():
            if not isinstance(key,str):
                raise TypeError('The keys of your dictionary should be of type string!')

        #I'm checking if the values of the inummyut dictionary is a numpy array or not    
        for value in data.values():
            if not isinstance(value,nummy.ndarray):
                raise TypeError('The values of your dictionary should be numpy arrays!')

        #I'm checking if the numpy array is one dimensional or not
        if value.ndim != 1 :
            raise TypeError('The dimension of your numpy array can not be more than 1 sweety!')

    #The numpy arrayd entered must have the same length
    def _check_array_lengths(self, data):
        for j,value in enumerate(data.values()):
            if j == 0:
                arrayLength = len(value)
            elif arrayLength != len(value):
                raise ValueError('All arrays must have the same length!')

    #Unicode type arrays are not flexible to work with,that's why we turn unicode type to object type
    def _convert_unicode_to_object(self, data):
        new_data = {}
        for key,value in data.items():
            if value.dtype.kind == 'U':
                new_data[key] = value.astype('object')
            else:
                new_data[key] = value
        return new_data

    #returns the number of rows in the dataframe
    def __len__(self):
        for value in self._data.values():
            return len(value)
        pass

    @property
    #user can get a list of columns by simply calling this method
    def columns(self):
        columnsList = []
        for key in self._data.keys():
            columnsList.append(key)
        return columnsList        

    #user must give a list of columns which have the same length
    @columns.setter
    def columns(self, columns):
        if not isinstance(columns,list):
            raise TypeError('You must provide columns as a list!')

        if len(self._data) != len(columns):
            raise ValueError('Length of columns list must be equal to the current dataframe!')

        for col in columns:
            if not isinstance(col,str):
                raise TypeError('Column name must be string sweetie!')

        if len(columns) != len(set(columns)):
            raise ValueError('Column names can not be duplicates!')
    
        #Now let's reassign the data after we have validated our info
        self._data = dict(zip(columns,self._data.values()))

    @property
    # gives u the number of columns and rows
    def shape(self):
        return len(self),len(self._data)

    #This method only functions in Ipython Jupyter Notebook to give you a pretty interface for your dataframe
    def _repr_html_(self):
        html = '<table style="border-collapse: collapse;margin: 25px 0;font-size: 0.9em;font-family: sans-serif;min-width: 400px;box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);"><thead style="background-color: #009879;color: #ffffff;text-align: left;"><tr style=" border-bottom: 1px solid #dddddd;"><th></th>'
        for col in self.columns:
            html += f"<th >{col:10}</th>"

        html += '</tr></thead>'
        html += "<tbody>"

        only_head = False
        num_head = 10
        num_tail = 10
        if len(self) <= 20:
            only_head = True
            num_head = len(self)

        for i in range(num_head):
            html += f'<tr><td><strong>{i}</strong></td>'
            for col, values in self._data.items():
                kind = values.dtype.kind
                if kind == 'f':
                    html += f'<td>{values[i]:10.3f}</td>'
                elif kind == 'b':
                    html += f'<td>{values[i]}</td>'
                elif kind == 'O':
                    v = values[i]
                    if v is None:
                        v = 'None'
                    html += f'<td>{v:10}</td>'
                else:
                    html += f'<td>{values[i]:10}</td>'
            html += '</tr>'

        if not only_head:
            html += '<tr><strong><td>...</td></strong>'
            for i in range(len(self.columns)):
                html += '<td>...</td>'
            html += '</tr>'
            for i in range(-num_tail, 0):
                html += f'<tr><td><strong>{len(self) + i}</strong></td>'
                for col, values in self._data.items():
                    kind = values.dtype.kind
                    if kind == 'f':
                        html += f'<td>{values[i]:10.3f}</td>'
                    elif kind == 'b':
                        html += f'<td>{values[i]}</td>'
                    elif kind == 'O':
                        v = values[i]
                        if v is None:
                            v = 'None'
                        html += f'<td>{v:10}</td>'
                    else:
                        html += f'<td>{values[i]:10}</td>'
                html += '</tr>'

        html += '</tbody></table>'
        return html

    #You can read CSV files by giving the path of your file to this function
    def read_csv_file_nila(self,path_of_file):
        with open(path_of_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                    line_count += 1
            print(f'Processed {line_count} lines.')

    #You can read text files by giving the path of your file to this function
    def read_txt_file_nila(self,path_of_file):
        with open(path_of_file) as f:
            lines = f.read()  
        print(lines)    


    def read_excel_file_nila(Self,path_of_file):
        workbook = xlrd.open_workbook(path_of_file)
        worksheet = workbook.sheet_by_index(0)
        for i in range(0, 5):
            for j in range(0, 3):
                print(worksheet.cell_value(i, j), end='\t')
            print('')

    @property
    #returns a NumPy array of data
    def values(self):
        return nummy.column_stack(list(self._data.values()))

    @property
    #This function gives you the column names and their corresponding data types
    def dtypes(self):
        typeList = []
        DTYPE_NAME = {'O': 'string', 'i': 'int', 'f': 'float', 'b': 'bool'}
        columnNames = nummy.array(list(self._data.keys()))
        for value in self._data.values():
            typeList.append(DTYPE_NAME[value.dtype.kind])
        typeList = nummy.array(typeList)
        new_data = {'Column Name' : columnNames,'Data Type' : typeList}
        return DataFrame(new_data)

    

    #Guys!Read the python.org documentation on Emulating numeric types and all the special methods and their corresponding operators are shown there.
    #We will simply be implementing those for arithmetic operations down below!
    
    def __sub__(self, other):
        return self._oper('__sub__', other)

    def __rsub__(self, other):
        return self._oper('__rsub__', other)

    def __mul__(self, other):
        return self._oper('__mul__', other)

    def __rmul__(self, other):
        return self._oper('__rmul__', other)

    def __add__(self, other):
            return self._oper('__add__', other)

    def __radd__(self, other):
        return self._oper('__radd__', other)

    def __floordiv__(self, other):
        return self._oper('__floordiv__', other)

    def __rfloordiv__(self, other):
        return self._oper('__rfloordiv__', other)

    def __truediv__(self, other):
        return self._oper('__truediv__', other)

    def __rtruediv__(self, other):
        return self._oper('__rtruediv__', other)

    def __pow__(self, other):
        return self._oper('__pow__', other)

    def __rpow__(self, other):
        return self._oper('__rpow__', other)

    def __gt__(self, other):
        return self._oper('__gt__', other)

    def __lt__(self, other):
        return self._oper('__lt__', other)

    def __ge__(self, other):
        return self._oper('__ge__', other)

    def __le__(self, other):
        return self._oper('__le__', other)

    def __ne__(self, other):
        return self._oper('__ne__', other)

    def __eq__(self, other):
        return self._oper('__eq__', other)

    #operator method
    def _oper(self, op, other):
        #if what you want to contain in your arithmatic operation is a dataframe then all you need to do is to check the dimension and see if it matches
        if isinstance(other, DataFrame):
            if other.shape[1] != 1:
                raise ValueError('`other` must be a one-column DataFrame.Single Dimension Guys')
            other = next(iter(other._data.values()))
        new_data = {}
        for col, values in self._data.items():
            #operation is given as a string .look above
            #getattr is a buil-in function that gives a method if u give it a string
            method = getattr(values, op)
            new_data[col] = method(other)
        return DataFrame(new_data)




    def __getitem__(self, object):
        
        #if you want to get a spicifc column pass it through your dataframe constructor 
        if isinstance(object,str):
            return DataFrame({object:self._data[object]})
    
        #if you want to select multiple columns of your dataframe the below comes in handy:)
        if isinstance(object,list):
            objlist= []
            return DataFrame({obj: self._data[obj] for obj in object})

        #if you wanted to select columns and rows simoltaniously
        if isinstance(object,tuple):
            return self._getitem_tuple(object)

        
        raise TypeError('You should give a string or list to get something back!')
