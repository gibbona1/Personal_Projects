import pandas as pd
import numpy as np
from docx import Document
from datetime import datetime
from dateutil.parser import parse
import sys

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        string = string.replace('.', ' ').replace('_', ' ').strip()
        sname_parse = parse(string, fuzzy=fuzzy)
        return True, sname_parse.strftime('%d.%m.%Y')
    except ValueError:
        if str_format is None:
            str_format = input("dateutil.parser did not work on sheet name, please input date format manually (e.g. '%d.%m.%Y')")
        else:
            print(f"using date format {str_format} to parse sheet name")
        try:
            sname_parse = datetime.strptime(string, str_format).strftime('%d.%m.%Y')
            return True, sname_parse
        except ValueError:
            print("could not parse any date format, setting blank")
            return False, ''

today_str = datetime.now().date().isoformat()

str_format = None

print(f"Using date {today_str}")

#excel_file = 'For Anthony.xlsx'
#filename   = "Register - GE Morning (4).docx"
excel_file = input("input excel file name (with .xlsx extension): ")
filename   = input("input word file name (with .docx extension): ")

try:
    xl = pd.ExcelFile(excel_file)

    sheetname = input(f"Choose sheet name from {xl.sheet_names}: ")

    names_df = pd.read_excel(excel_file, sheet_name = sheetname)
    names_df = names_df.iloc[:, :4]
    names_df.columns = ["Level", "Name", "Start", "End"]
    for c in ['Level', 'Start']: #names_df.columns:
        names_df[c] = names_df[c].replace('\n', ' ', regex=True)
        names_df[c] = names_df[c].str.strip()
    t_starts = np.where(names_df.Start == "Course Start Date")[0]
    print(f"There are {len(t_starts)} tables to fill")
except Exception as e:
    print("Error while reading the excel")
    print(e)
    input("Press Enter to Exit.")
    sys.exit()

def fill_template(class_i):
    doc = Document(filename)

    if class_i == len(t_starts) - 1:
        tmp_tab = names_df.iloc[t_starts[class_i]:]
    else:
        tmp_tab = names_df.iloc[t_starts[class_i]:t_starts[class_i+1]]

    tmp_tab     = tmp_tab.dropna(subset = ['Name'])
    group_level = tmp_tab.Level.values[0]
    #print("room_str", tmp_tab.Level.values[1])
    room_list = tmp_tab.Level.values[1].replace('  ', ' ').replace(' - ', '-').split(' ')[3:]
    #print("room_list", room_list)
    group_room  = ''.join(room_list)

    teacher_sub = tmp_tab[tmp_tab.Start.str.replace('\n', ' ') == "Course Start Date"].Name.values[0]
    teacher_sub = [ts.strip() for ts in teacher_sub.strip().split('\n')[:2]]
    if len(teacher_sub) < 2:
        teacher_sub.extend(' ')
    student_tab = tmp_tab.iloc[1:]

    attend_tab = doc.tables[0]
    prev_text = '<>'
    level_str = 'CEFR level:'
    level_str2 = 'Week beginning:'
    for i, row in enumerate(attend_tab.rows[1:]):
            for j, cell in enumerate(row.cells):
                if cell.text:
                    if prev_text == level_str and cell.text != level_str:
                        print(f"    changing {level_str[:-1]} to {group_level}")
                        cell.text = group_level
                    if prev_text == level_str2 and cell.text != level_str2:
                        sname_isdate, sname_parse = is_date(sheetname)
                        cell.text = sname_parse
                        if sname_isdate:
                            print(f"    changing {level_str2[:-1]} to {sname_parse}")
                        else:
                            print(f"    changing {level_str2[:-1]} to blank since sheet name is not a date")
                    if cell.text == 'Tmp_teacher_1':
                        cell.text = teacher_sub[0]
                    if cell.text == 'Tmp_teacher_2':
                        cell.text = teacher_sub[1]
                    prev_text = cell.text
    
    for i, row in enumerate(attend_tab.rows[5:]):
        for j, cell in enumerate(row.cells):
            if j == 1 and i in range(len(student_tab)):
                cell.text = student_tab.Name.values[i]
    print(f"    pasted {len(student_tab)} students to the table")
    save_str = f'{today_str}_{group_level}_{group_room}_Register.docx'
    save_str = save_str.replace('/', '-')
    doc.save(save_str)
    print(f'saved class {class_i+1} to {save_str}')

for i in range(len(t_starts)):
    print(f'processing class {i+1} of {len(t_starts)}.')
    try:
        fill_template(i)
    except Exception as e:
        print(f"Error running the filling the word template")
        print(e)
        input("Press Enter to Exit.")
        sys.exit()
        

input("Done! Press Enter to Exit.")