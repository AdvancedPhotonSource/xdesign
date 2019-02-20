#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import requests

# TODO: Add command line interface to download profiles for all of the compounds
# in the NIST database

# List all available elements and materials.
url = "http://xrayplots.2mrd.com.au/api/getall"

# Fetch data from website
response = requests.get(url)

# Format data as JSON
jsondata = response.json()

# Reorganize data by name
newjsondata = dict()
for point in jsondata:
    name = point.pop('name')

    # Remove leading and ending white space
    point['symbol'] = point['symbol'].strip()
    point['z'] = point['z'].strip()
    point['density'] = float(point['density'])

    newjsondata[name.strip()] = point

# Save the data to a file
with open("NIST_index.json", "w", encoding="utf-8") as f:
    json.dump(newjsondata, f)
