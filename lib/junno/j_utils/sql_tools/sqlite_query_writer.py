

class SQLQuery:
    """
    DOCUMENT THIS CLASS
    """
    def __init__(self, query_type, **kwargs):

        self.table_name = kwargs.get('table_or_subquery', None)
        self.column_name = kwargs.get('column_name', None)

        if query_type is "Select" or query_type is "Update":
            self.where = kwargs.get('where', None)

        if query_type is "Update" or query_type is "Insert":

            self.update_value = kwargs.get("update_value", None)
            # Options if update not available
            self.rollback = kwargs.get('rollback', False)
            self.abort = kwargs.get('abort', False)
            self.replace = kwargs.get('replace', False)
            self.fail = kwargs.get('fail', False)
            self.ignore = kwargs.get('ignore', False)
            self.list_options = [self.rollback, self.abort, self.replace, self.fail, self.ignore]
            try:
                assert (sum(self.list_options) == 1 or
                        sum(self.list_options) == 0)
            except AssertionError:
                print("You can't only have one (or zero) alternative to UPDATE statement. Disabling all of them.")
                self.rollback = False
                self.abort = False
                self.replace = False
                self.fail = False
                self.ignore = False

        if query_type is "Insert" or query_type is "CreateTable":
            self.select_statement = kwargs.get('select_statement', None)
            self.schema_name = kwargs.get('schema_name', None)

    def cast_list2str(self, list_var):
        if isinstance(list_var, (list, tuple)):
            list_var = [str(i) for i in list_var]
            return ','.join(list_var)
        elif isinstance(list_var, str):
            return list_var
        else:
            raise NotImplementedError('Cannot convert input into string')

    def cast_str2list(self, str_var):
        if isinstance(str_var, (list, tuple)):
            return str_var
        elif isinstance(str_var, str):
            return str_var.split(',')
        else:
            raise NotImplementedError('Cannot convert input into list')

    @property
    def where(self):
        return self._where

    @where.setter
    def where(self, where):
        if where is not None:
            try:
                assert (isinstance(where, str))
                self._where = where
            except AssertionError:
                print("WHERE clause must be a string. Otherwise, it's ignored and casted to None")
                self._where = None
        else:
            self._where = None

    @property
    def column_list_names(self):
        return self.cast_str2list(self.column_name)

    @property
    def simple_table_name(self):
        return self.table_name.split(' ')[0]

class CustomQuery(SQLQuery):
    """
    Write your own SQL Query. You need also to precise if it's a reader or writer query
    """
    def __init__(self, query, action):
        """

        :param query: A string representing the query
        :param action: reader or writer
        """
        self.query = query
        self.action = action

    def __str__(self):
        return str(self.query)

class SelectQuery(SQLQuery):
    """
    This only  represents the select-core
    """
    def __init__(self, **kwargs):

        SQLQuery.__init__(self, query_type="Select", **kwargs)

        self.distinct = kwargs.get('distinct', False)
        self.all = kwargs.get('all', False)
        self.group_by = kwargs.get('group_by', None)
        self.having = kwargs.get('having', None)
        self.ordering_term = kwargs.get('ordering_term', None)
        self.ordering_ascent = kwargs.get('ordering_ascent', True)
        self.limit = kwargs.get('limit', None)
        self.offset = kwargs.get('offset', None)
        self.action = "reader"

    @staticmethod
    def fromStr(query):
        words = query.split(' ')
        if not words or words[0].lower() != 'select' or len(words) < 2:
            return None
        words = words[1:]
        kwargs = {}
        if words[0].lower() == 'or':
            kwargs[words[1].lower()] = True
            words = words[2:]

        attribute_name = 'column_name'
        attribute_content = ['']

        def apply_attribute():
            if attribute_content[-1] == '':
                del attribute_content[-1]
            if attribute_content:
                content = attribute_content if len(attribute_content)>1 else attribute_content[0]
                kwargs[attribute_name] = content

        previous_word = ""
        for word in words:
            if word.lower() in ('from', 'limit', 'offset', 'order', 'group', 'having', 'where'):
                apply_attribute()
                attribute_content = ['']
                attribute_name = word.lower()
                if word.lower() == 'from':
                    attribute_name = 'table_or_subquery'
                elif word.lower() == 'group':
                    attribute_name = 'group_by'
                elif word.lower() == 'order':
                    attribute_name = 'ordering_term'
            elif word.lower() == 'by' and previous_word in ('order', 'group'):
                continue
            elif (word.lower() == 'asc' or word.lower() == 'desc') and attribute_name == 'ordering_term':
                kwargs['ordering_ascent'] = word.lower() == 'asc'
            else:
                if attribute_content[-1]:
                    attribute_content[-1] += ' '
                if word[-1] == ',':
                    attribute_content[-1] += word[:-1]
                    attribute_content.append('')
                else:
                    attribute_content[-1] += word
            previous_word = word
        apply_attribute()
        return SelectQuery(**kwargs)

    @property
    def having(self):
        return self._having

    @having.setter
    def having(self, having):
        if having is not None:
            try:
                assert(isinstance(having, str))
                self._having = having
            except AssertionError:
                print("Having clause must be a string. Otherwise, it's ignored and casted to None")
                self._having = None

    @property
    def group_by(self):
        return self._group_by

    @group_by.setter
    def group_by(self, group_by):
        if group_by is not None:
            self._group_by = self.cast_list2str(group_by)

        else:
            self._group_by = None

    @property
    def ordering_term(self):
        return self._ordering_term

    @ordering_term.setter
    def ordering_term(self, ordering_term):
        if ordering_term is not None:
            self._ordering_term = self.cast_list2str(ordering_term)
        else:
            self._ordering_term = None

    @property
    def ordering_ascent(self):
        return self._ordering_ascent

    @ordering_ascent.setter
    def ordering_ascent(self, ordering_ascent):
        if ordering_ascent is not None:
            self._ordering_ascent = ordering_ascent
        else:
            self._ordering_ascent = True

    @property
    def column_name(self):
        return self._column_name

    @column_name.setter
    def column_name(self, column_name):
        if column_name is not None:
            self._column_name = self.cast_list2str(column_name)
        else:
            raise ValueError("'SelectQuery' is expecting a not None argument of the form column_name=list or string.")

    def __str__(self):
        query = "SELECT "
        if self.distinct:
            query += "DISTINCT "
            try:
                assert(not self.all)
            except AssertionError:
                print("You can't have both a DISTINCT and ALL clause. Ignoring ALL clause")
                self.all = False

        if self.all:
            query += "ALL "

        query += self.column_name

        if self.table_name is not None:
            query += " FROM " + self.table_name

        if self.where is not None:
            query += " WHERE ("+self.where + ")"

        if self.group_by is not None:
            query += " GROUP BY " + self.group_by
            if self.having is not None:
                query += " HAVING " + self.having

        if self.ordering_term is not None:
            query += " ORDER BY " + self.ordering_term + (' ASC' if self.ordering_ascent else ' DESC')

        if self.limit is not None:
            query += " LIMIT " + str(self.limit)
            if self.offset is not None:
                query += " OFFSET " + str(self.offset)

        return query


class UpdateQuery(SQLQuery):
    """
    Lexical implementation of the update-stmt:
    https://www.sqlite.org/lang_update.html
    """
    def __init__(self, **kwargs):

        SQLQuery.__init__(self, query_type="Update", **kwargs)

        self.action = "writer"

    @property
    def column_name(self):
        return self._column_name

    @column_name.setter
    def column_name(self, column_name):
        if column_name is not None:
            self._column_name = self.cast_str2list(column_name)
            if all(['=' in x for x in self.column_name]):
                self.update_value = None
        else:
            raise ValueError("'UpdateQuery' is expecting a not None argument of the form column_name=list or string.")

    @property
    def update_value(self):
        return self._update_value

    @update_value.setter
    def update_value(self, update_value):
        if update_value is not None:
            self._update_value = self.cast_str2list(update_value)
            try:
                assert(len(self.update_value) == len(self.column_name))
            except AssertionError:
                print("Dimensions not matching between attribute column_name of length: ", len(self.column_name))
                print("And attribute update_value of length: ", len(self.update_value))
                raise SystemExit
        elif all(['=' in x for x in self.column_name]):
            self._update_value = None
        else:
            raise ValueError("'UpdateQuery' is expecting a not None argument of the form update_value=list or string.")

    def __str__(self):
        query = "UPDATE "
        if self.rollback:
            query += "OR ROLLBACK "
        elif self.abort:
            query += "OR ABORT "
        elif self.replace:
            query += "OR REPlACE "
        elif self.fail:
            query += "OR FAIL "
        elif self.ignore:
            query += "OR IGNORE "

        query += self.table_name + " SET "

        if self.update_value is not None:
            for i, element in enumerate(self.column_name):
                query += self.column_name[i]+'='+str(self.update_value[i])+', '
            query = query[:-2] # Remove the last character (useless comma)
        else:
            query += self.cast_list2str(self.column_name)

        if self.where is not None:
            query += " WHERE ("+self.where + ")"

        return query


class InsertQuery(SQLQuery):
    """
    Lexical implementation of the insert-stmt:
    https://www.sqlite.org/lang_insert.html
    """
    def __init__(self, **kwargs):

        SQLQuery.__init__(self, query_type="Insert", **kwargs)

        self.total_replace = kwargs.get('total_replace', False)

        self.default_value = kwargs.get('default_value', False)

        self.action = "writer"

    @property
    def column_name(self):
        return self._column_name

    @column_name.setter
    def column_name(self, column_name):
        if column_name is not None:
            self._column_name = self.cast_str2list(column_name)
        else:
            self._column_name = None

    @property
    def update_value(self):
        return self._update_value

    @update_value.setter
    def update_value(self, update_value):
        if update_value is not None:
            self._update_value = self.cast_str2list(update_value)
            if self.column_name is not None:
                try:
                    assert (len(self.update_value) == len(self.column_name))
                except AssertionError:
                    print("Dimensions not matching between attribute column_name of length: ", len(self.column_name))
                    print("And attribute update_value of length: ", len(self.update_value))
                    raise SystemExit
        else:
            self._update_value = None

    def __str__(self):
        if not self.total_replace:
            query = "INSERT "
        else:
            query = "REPLACE "

        if self.rollback:
            query += "OR ROLLBACK "
        elif self.abort:
            query += "OR ABORT "
        elif self.replace:
            query += "OR REPlACE "
        elif self.fail:
            query += "OR FAIL "
        elif self.ignore:
            query += "OR IGNORE "

        query += "INTO "
        if self.schema_name is not None:
            query += self.schema_name+'.'

        query += self.table_name

        if self.column_name is not None:
            query += "("+self.cast_list2str(self.column_name)+") "

        if self.update_value is not None:
            query += "VALUES (" + self.cast_list2str(self.update_value) + ")"

        elif self.select_statement is not None:
            query += " "+str(self.select_statement)

        elif self.default_value:

            query += "DEFAULT VALUES"

        return query


class CreateTableQuery(SQLQuery):
    """
    Lexical implementation of the insert-stmt:
    https://www.sqlite.org/lang_createtable.html
    """
    def __init__(self, **kwargs):
        SQLQuery.__init__(self, query_type="CreateTable", **kwargs)

        self.temp = kwargs.get('temporary', False)+kwargs.get('temp', False)

        self.if_not_exists = kwargs.get('if_not_exists', False)

        self.table_constraint = kwargs.get('table_constraint', None)

        self.column_constraint = kwargs.get('column_constraint', None)

        self.without_rowid = kwargs.get('without_rowid', False)

        self.action = "writer"

    @property
    def column_constraint(self):
        return self._column_constraint

    @column_constraint.setter
    def column_constraint(self, column_constraint):
        if self.column_name is not None:
            if isinstance(column_constraint, list):
                self._column_constraint = column_constraint
            elif isinstance(column_constraint, str):
                self._column_constraint = self.cast_str2list(column_constraint)

            try:
                assert(len(self._column_constraint) <= len(self.column_name))
                self._column_constraint += [None]*(len(self.column_name) - len(self._column_constraint))
            except AssertionError:
                print("Dimensions not matching between attribute column_name of length: ", len(self.column_name))
                print("And attribute column_constraint of length: ", len(self._column_constraint))
                raise SystemExit
        elif column_constraint is None:
            self._column_constraint = None

    @property
    def table_constraint(self):
        return self._table_constraint

    @table_constraint.setter
    def table_constraint(self, table_constraint):
        if table_constraint is not None:
            self._table_constraint = self.cast_str2list(table_constraint)
        else:
            self._table_constraint = None

    def __str__(self):
        query = "CREATE "

        if self.temp:
            query += "TEMP "
        query += "TABLE "

        if self.if_not_exists:
            query += "IF NOT EXISTS "

        if self.schema_name is not None:
            query += self.schema_name+'.'

        query += self.table_name

        if self.column_name is not None:
            query += "("
            for index, element in enumerate(self.column_name):
                query += element
                column_constraint = self.column_constraint[index]
                if column_constraint is not None:
                    query += ' ' + column_constraint
                query += ', '
            query = query[:-2]
            if self.table_constraint is not None:
                query += ", " + self.cast_list2str(self.table_constraint)
            query += ") "

            if self.without_rowid:
                query += "WITHOUT ROWID"
        else:
            query += "AS " + str(self.select_statement)

        return query



















