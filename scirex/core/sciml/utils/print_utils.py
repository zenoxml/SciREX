# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and AiREX Lab,
# Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# SciREX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SciREX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with SciREX. If not, see <https://www.gnu.org/licenses/>.
#
# For any clarifications or special considerations,
# please contact <scirex@zenteiq.ai>
# Author: Thivin Anandh D

from rich.console import Console
from rich.table import Table


def print_table(title, columns, col_1_values, col_2_values):
    """
    This function prints a table with two columns to the console.

    :param title: Title of the table
    :type title: str
    :param columns: List of column names
    :type columns: list
    :param col_1_values: List of values for column 1
    :type col_1_values: list
    :param col_2_values: List of values for column 2
    :type col_2_values: list

    :return: None
    :rtype: None
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
