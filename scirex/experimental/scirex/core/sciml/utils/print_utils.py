# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org


"""
File : print_utils.py
Purpose: This file contains utility functions to print tables to the console.

Functions:
    - print_table: Print a table with two columns to the console

Authors:
    Thivin Anandh D (https://thivinanandh.github.io)

Version Info:
    27/Dec/2024: Initial version - Thivin Anandh D
"""

from rich.console import Console
from rich.table import Table


def print_table(title: str, columns: list, col_1_values: list, col_2_values: list):
    """
    This function prints a table with two columns to the console.

    Args:
        title: str: Title of the table
        columns: list: List of column names
        col_1_values: list: List of values for column 1
        col_2_values: list: List of values for column

    Returns:
        None
    """

    # Create a console object
    console = Console()

    # Create a table with a title
    table = Table(show_header=True, header_style="bold magenta", title=title)

    # Add columns to the table
    for column in columns:
        table.add_column(column)

    # Add rows to the table
    for val_1, val_2 in zip(col_1_values, col_2_values):
        # check if val_2 is a float
        if isinstance(val_2, float):
            # add the row to the table
            table.add_row(val_1, f"{val_2:.4f}")
        else:
            table.add_row(val_1, str(val_2))

    # Print the table to the console
    console.print(table)
