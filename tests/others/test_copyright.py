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

# Author: Thivin Anandh D (https://thivinanandh.github.io)
# Version Info: 06/Jan/2025: Initial version - Thivin Anandh D

from pathlib import Path
import pytest


def test_copyright():
    """Test if all Python files have the required copyright header."""
    # Get root directory path
    root_dir = Path(__file__).parent.parent

    # Get copyright header content
    try:
        with open(root_dir / "CopyrightHeader.txt", "r") as f:
            copyright_header = f.read().strip()
            # print("Header:", repr(copyright_header))
    except FileNotFoundError:
        pytest.fail("Copyright header template file not found")

    # Get all Python files
    python_files = []
    scan_dirs = ["scirex", "tests", "examples"]

    for dir_name in scan_dirs:
        dir_path = root_dir / dir_name
        if dir_path.exists():
            python_files.extend(dir_path.rglob("*.py"))

    # remoce all __init__.py files
    python_files = [
        file_path for file_path in python_files if "__init__.py" not in str(file_path)
    ]

    # Check each file for copyright header
    files_missing_header = []
    for file_path in python_files:
        try:
            with open(file_path, "r") as f:
                content = f.read()
                if copyright_header not in content:
                    files_missing_header.append(str(file_path))
        except Exception as e:
            pytest.fail(f"Error reading file {file_path}: {str(e)}")

    if files_missing_header:
        files_list = "\n".join(files_missing_header)
        print(f"The following files are missing the copyright header:\n{files_list}")
        pytest.fail(
            f"The following files are missing the copyright header:\n{files_list}"
        )


if __name__ == "__main__":
    test_copyright()
