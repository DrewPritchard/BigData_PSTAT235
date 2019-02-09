import zipfile
zip_ref = zipfile.ZipFile("two-sigma-connect-rental-listing-inquiries.zip", 'r')
zip_ref.extractall("data/")
zip_ref.close()