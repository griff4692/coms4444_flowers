#!/usr/bin/env python
# coding:utf-8

"""Recursively find files and prepend license and author information to each file.
Usage:
    1. Prepare license file and authors file in advance.
        (Specified by license_text_file and authors_yaml_file, respectively)
    2. Change hard-coded parameters in this script:
        * input directory
        * output directory
        * exclude list with exact name
        * exclude list with regular expression
        * shebang 
        * license file
        * authors file
        * whether if copy binary files and excluded files from input directory to output directory
    3. Run this script without arguments.
Note:
    * empty files are copied to output directory without prepending
"""
import os
import re
import shutil
import sys
import yaml


def get_file_list(directory):
    directory_list = os.walk(directory)
    root_and_file_list = []
    for (root, _, files) in directory_list:
        file_list = []  # for each root
        for fname in files:
            file_list.append(fname)
        root_and_file_list.append([root, file_list])
    return root_and_file_list


def check_exclusion(fname, regex_list, name_list):
    for name_item in name_list:
        if name_item in fname:
            return True
    for regex_item in regex_list:
        if regex_item.search(fname) is not None:
            return True
    return False


# detect script/file type and set proper comment out type
def comment_out_type(fname, interpreter):
    start = '\"\"\"'  # python for default
    end = '\"\"\"'  # python for default
    each = ''  # python for default
    if (re.search(r'\.sh$', fname) is not None) or (re.search(r'\.bash$', fname) is not None) or (
            'sh' in interpreter) or ('bash' in interpreter):  #bash
        start = ": '"
        end = "'"
        each = ''
    elif (re.search(r'\.c$', fname) is not None) or (re.search(r'\.h$', fname) is not None):  # clang
        start = '/*'
        end = ' */'
        each = ''
    elif (re.search(r'\.R$', fname) is not None) or (re.search(r'\.r$', fname) is not None):  # R?
        start = ''
        end = ''
        each = '#'
    elif re.search(r'\.m$', fname) is not None:  # matlab?
        start = '%{'
        end = '%}'
        each = ''
    elif re.search(r'\.java$', fname) is not None:  # Java
        start = '/*'
        end = ' */'
        each = ''
    elif (re.search(r'\.md$', fname) is not None) or (re.search(r'\.html$', fname) is not None) or (
            re.search(r'\.htm$', fname) is not None):  # markdown or html
        start = '<!--- '
        end = ' --->'
        each = ''
    #assert (len(start) + len(end) == 0) ^ (
    #            len(each) == 0), '[EE] error in method comment_out_type. Recheck comment out type.'
    return (start, end, each)


if __name__ == '__main__':

    # Parameters
    input_directory = './indir/'
    output_directory = './outdir/'
    #
    exclude_list_regex = [
        r'\.git', r'TODO', r'AUTHORS', r'MANIFEST\.in', r'README\.md', r'LICENSE', r'doxygen\.ini',
        r'\.cls$', r'\.bib$', r'\.sty$', r'\.tex$', r'\.yml$', r'\.xml$', r'\.bib$', r'\.gz$',
        r'\.txt$', r'\.dat$', r'\.adat$', r'\.tdat$', r'\.info$', r'code-postprocessing/cocopp/tth'
    ]  # exclude list with regular expression
    exclude_list_name = [
        'code-experiments/build/python/cython/interface.c',  # for example
        'code-experiments/test/unit-test/test_hypervolume.txt'  # for example
    ]  # exclude list with exact name
    #
    shebangs = []
    shebangs.append(r'^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)')  # shebang such as # -*- coding: utf-8 -*-
    shebangs.append(r'^[ \t\f]*#.*?mode[:=][ \t]*([-_.a-zA-Z0-9]+)')  # shebang such as # -*- mode: cython -*-
    shebangs.append(r'^#cython:')  # shebang for cython?
    #
    license_text_file = 'license.txt'
    authors_yaml_file = 'list_author.yml'
    exception_file_key = 'AUTHOR_FOR_EXCEPTION_FILE'
    #
    if_copy_not_changed_file = False  # whether if copy the unchanged files from input_dir to output_dir

    # Complie regexes
    shebang_interpreter_compiled = re.compile(r'^#!')  # shebang such as #!/usr/bin/env python
    shebangs_comipled = [re.compile(i) for i in shebangs]
    exclude_list_regex_compiled = [re.compile(i) for i in exclude_list_regex]

    # Directory and file handling
    if not input_directory.endswith(os.sep):
        input_directory += os.sep
    if not output_directory.endswith(os.sep):
        output_directory += os.sep

    if not os.path.exists(input_directory):
        print '[EE] directory ' + input_directory + ' does not exist.'
        sys.exit()
    if not os.path.isdir(input_directory):
        print '[EE] directory ' + input_directory + ' is not a directory.'
        sys.exit()

    if os.path.isdir(output_directory):
        if not os.listdir(output_directory):
            print '[EE] directory ' + output_directory + 'already exists and not empty. Exit problem to avoid from ' \
                                                   'mixing old existing files and new files. '
            sys.exit()
    else:
        try:
            os.mkdir(output_directory)
        except:
            print '[EE] directory ' + output_directory + ' cannot be created.'
            sys.exit()

    try:
        with open(license_text_file, 'r') as license_file:
            license_content = license_file.read()
    except:
        print '[EE] license information text file ' + license_text_file + ' is missing. Please prepare it in advance.'
        sys.exit()

    try:
        with open(authors_yaml_file, 'r') as yaml_file:
            yaml_object = yaml.load(stream=yaml_file, Loader=yaml.SafeLoader)
    except:
        print '[EE] author information yaml file ' + authors_yaml_file + ' is missing. Please prepare it in advance.'
        sys.exit()

    # Obtain file list
    target_root_and_file_list = get_file_list(input_directory)

    # Prepend header for each target file
    for (root, files) in target_root_and_file_list:
        relative_output_directory = output_directory + root[len(input_directory):] + os.sep

        # Makedir for each root directory
        if not os.path.exists(relative_output_directory):
            try:
                os.makedirs(relative_output_directory)
            except:
                print '[EE] directory ' + relative_output_directory + ' cannot be created.'
                sys.exit()

        for fname in files:
            target_input_file = os.path.join(root, fname)
            target_output_file = relative_output_directory + fname

            # remove heading input_directory from target_input_file so as to match key
            name_body = target_input_file[len(input_directory):]

            is_excluded = check_exclusion(name_body, exclude_list_regex_compiled, exclude_list_name)
            is_binary = False
            if not is_excluded:
                with open(target_input_file, 'r') as input_file:
                    # Detect shebang
                    interpreter = ''
                    shebang_line = 0
                    for shebang_line, line in enumerate(input_file):
                        conditionI = shebang_interpreter_compiled.search(line) is None
                        conditions = [i.search(line) is None for i in shebangs_comipled]
                        if not conditionI:  # store interpreter shebang info for determining comment out style
                            interpreter = shebang_interpreter_compiled.search(line).group(0)
                        if all(conditions):  # no match for any shebang = normal line has started
                            break
                        if shebang_line > 2:  # shebang should be on the first or second line of the file
                            break
                    input_file.seek(0)  # rewind

                    # Prepare header string for each file
                    key = name_body
                    authors = yaml_object.get(key, yaml_object.get(exception_file_key, []))  # if KeyError, default key (exception_file_key) is obtained
                    (comment_out_start, comment_out_end, comment_out_each) = comment_out_type(target_input_file,
                                                                                              interpreter)
                    buffer_header = []
                    buffer_header.append('{0}{1}\n'.format(comment_out_each, comment_out_start))
                    buffer_header.append('{0}{1}\n'.format(comment_out_each, license_content))
                    if authors:
                        buffer_header.append('{0}    AUTHORS:\n'.format(comment_out_each))
                        buffer_header.append('{0}      '.format(comment_out_each))
                        for author in authors:
                            buffer_header.append('  {0}'.format(author))
                    buffer_header.append('\n{0}{1}\n\n'.format(comment_out_each, comment_out_end))
                    header = ''.join(buffer_header)

                    # Do prepending
                    buffer_file_content = []
                    for index, line in enumerate(input_file):
                        # Detect whether the file is binary/ascii during the read
                        try:
                            line.decode('ascii')
                        except UnicodeDecodeError:
                            is_binary = True
                            break
                        else:
                            if index == shebang_line:
                                buffer_file_content.append(header)
                            buffer_file_content.append(line)
                    if not is_binary:
                        with open(target_output_file, 'w') as output_file:
                            output_file.write(''.join(buffer_file_content))
                            print '[II] success: {0}'.format(target_output_file)

            if if_copy_not_changed_file:
                if is_excluded or is_binary:  # just copy from input_directory to output_directory
                    try:
                        shutil.copy2(target_input_file, target_output_file)
                        reason = 'it matches to an item in the exclude list' if is_excluded else 'it is a non-ascii file'
                        print '[II] just copied {0} from {1} to {2}, since {3}.'.format(
                            name_body, input_directory, output_directory, reason)
                    except:
                        print '[WW] {0} is not changed nor copied.'.format(name_body)

