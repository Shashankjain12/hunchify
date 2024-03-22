#!/bin/bash

# Step 1: Identify added notebook files
added_notebooks=$(git diff --name-only --cached | grep '\.ipynb$')

# Step 2: Execute notebooks and export to HTML
for notebook in $added_notebooks; do
    jupyter nbconvert --to html --execute "$notebook"
done

# Step 3: Strip output from HTML files
for html_file in *.html; do
    # Use a tool like sed to remove output cells
    sed -i '/<div class="output_area">/,/<\/div>/d' "$html_file"
done