import sqlite3
import types
import logging
import sys
import numpy as np
import io
from time import sleep
from . import sqlite_query_writer as sql_w

LOG_FILENAME = 'log/sql_access.log'
LEVELS = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL}
if len(sys.argv) > 1:
    level_name = sys.argv[1]
    level = LEVELS.get(level_name, logging.NOTSET)
    if level != logging.NOTSET:
        logging.basicConfig(filenam=LOG_FILENAME, level=level)


class SQLAccessor:
    def __init__(self, database_path):
        print('Opening database at ', database_path)
        self.db = database_path

    def read_db(self, command, safe_argument=None, auto_retry=True):
        generator = self.read_db_generator(command, safe_argument=safe_argument, auto_retry=auto_retry)
        return list(generator)

    def read_db_generator(self, command, safe_argument=None, auto_retry=True):
        sqlite3.register_converter("array", self.convert_array)
        conn = sqlite3.connect(self.db, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        logging.info(str(command))
        if safe_argument is None:
            done = False
            while not done:
                try:
                    cur.execute(str(command))
                    done = True
                except sqlite3.OperationalError as err:
                    if auto_retry:
                        print(err)
                        sleep(3)
                    else:
                        raise err
        else:
            if isinstance(safe_argument, (list, types.GeneratorType)):
                cur.executemany(str(command), safe_argument)
            else:
                done = False
                while not done:
                    try:
                        cur.execute(str(command))
                        done = True
                    except sqlite3.OperationalError as err:
                        if auto_retry:
                            print(err)
                            sleep(3)
                        else:
                            raise err

        fetched_data = cur.fetchone()
        while fetched_data is not None:
            yield fetched_data
            fetched_data = cur.fetchone()

    def write_db(self, command, safe_argument=None):

        conn = sqlite3.connect(self.db, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        logging.info(str(command))
        if safe_argument is None:
            cur.execute(str(command))
        else:
            if isinstance(safe_argument, (list, types.GeneratorType)):
                cur.executemany(str(command), safe_argument)
            else:
                cur.execute(str(command), safe_argument)

        conn.commit()

    def execute_query(self, command, safe_argument=None):
        if command.action is "writer":
            sqlite3.register_adapter(np.ndarray, self.adapt_array)
            self.write_db(command, safe_argument)

        elif command.action is "reader":
            return self.read_db(command, safe_argument)

    def get_columm_name(self, table):
        command = "PRAGMA table_info("+table+")"
        info = self.read_db(command=command)
        return [_[1] for _ in info]

    def get_column_type(self, table):
        command = "PRAGMA table_info(" + table + ")"
        info = self.read_db(command=command)
        print(info)
        return [_[2] for _ in info]

    def drop_column(self, table, column_name):
        list_column = self.get_columm_name(table)
        if isinstance(column_name, str):
            column_name = [column_name]

        checkifpresent = [i for i, _ in enumerate(column_name) if _ in list_column ]
        column2drop = np.asarray(column_name)[checkifpresent]


        indices2keep = [i for i, _ in enumerate(list_column) if _ not in column2drop]
        columntype2keep = list(np.asarray(self.get_column_type(table))[indices2keep])

        if len(column2drop) == 0:
            print("Columns are already absent in the table", column_name)
            return None

        print("Dropping column(s): ", ", ".join(column2drop))
        column2keep = [_ for _ in list_column if _ not  in column2drop]

        command = [sql_w.CreateTableQuery(table_or_subquery ="t_backup", column_constraint=columntype2keep, column_name=column2keep),
                   sql_w.InsertQuery(table_or_subquery="t_backup", select_statement=sql_w.SelectQuery(column_name=column2keep, table_or_subquery=table)),
                   sql_w.CustomQuery(query="DROP TABLE "+table, action="writer"),
                   sql_w.CustomQuery("ALTER TABLE t_backup RENAME TO "+table, action="writer")]

        for _ in command:
            print(_)
            self.execute_query(_)


    def add_column(self, table, column_name, column_type):
        if isinstance(column_name, list):
            assert(len(column_name)==len(column_type))
            for a,b in zip(column_name, column_type):
                self.add_column(table, a, b)
        else:
            self.execute_query(sql_w.CustomQuery('ALTER TABLE '+table+' ADD COLUMN ' + column_name +" "+ column_type, action="writer"))



    def adapt_array(self, arr):
        """
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    def table_primary_key(self, table):
        for col_def in self.read_db("PRAGMA table_info(%s);" % table):
            id, name, type, is_nullable, default_value, is_pk = col_def
            if is_pk:
                return name
        return 'rowid'