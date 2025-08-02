#!/bin/sh

# Exit immediately if a command fails
set -e

rm -rf doxygen
mkdir doxygen

# generate a Doxygen config file
cd doxygen && doxygen -g

echo "PROJECT_NAME = \"Local Information Time Evolution (LITE)\"" > Doxyfile
echo "INPUT = ../" >> Doxyfile
echo "FILE_PATTERNS = *.md" >> Doxyfile
echo "MARKDOWN_SUPPORT = YES" >> Doxyfile
echo "USE_MDFILE_AS_MAINPAGE = ../introduction.md" >> Doxyfile
echo "EXTRACT_ALL = YES" >> Doxyfile
echo "RECURSIVE = YES" >> Doxyfile
echo "GENERATE_LATEX = YES" >> Doxyfile
echo "GENERATE_TREEVIEW = YES" >> Doxyfile
echo "USE_MATHJAX = YES" >> Doxyfile
echo "HTML_EXTRA_STYLESHEET = ../doxygen-awesome.css" >> Doxyfile

# Run doxygen to generate docs
echo "Generating Doxygen documentation..."
doxygen


# Check if HTML docs were generated successfully
if [ -d "html" ] && [ -f "html/index.html" ]; then
    echo "Documentation generated successfully."
else
    echo "Documentation generation failed or html output not found."
    exit 1
fi

mv html ../
cd ..
rm -rf doxygen