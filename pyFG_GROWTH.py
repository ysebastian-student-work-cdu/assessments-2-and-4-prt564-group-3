# Sample code to do FP-Growth in Python



# Sample code to do FP-Growth in Python


import pyfpgrowth

# Creating Sample Transactions
transactions = [

    ['Email addresses', 'IP addresses', 'Names', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Device information', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Names', 'Passwords'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses'],
    ['Email addresses', 'Employers', 'Job titles', 'Names', 'Phone numbers', 'Physical addresses',
     'Social media profiles'],

    ['Email addresses', 'Genders', 'Geographic locations', 'Marital statuses', 'Names', 'Passwords', 'Phone numbers'],
    ['Email addresses', 'Genders', 'Income levels', 'Job titles', 'Marital statuses', 'Names', 'Passwords',
     'Phone numbers', 'Physical addresses', 'Purchases', 'Religions', 'Salutations'],

    ['Email addresses', 'Password hints', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Races',
     'Relationship statuses', 'Sexual orientations', 'Spoken languages', 'Usernames'],

    ['Email addresses', 'Passwords', 'Spoken languages', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Names', 'Passwords'],
    ['Email addresses', 'Employers', 'Job titles', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Partial dates of birth', 'Passwords',
     'Usernames', 'Website activity'],

    ['Address book contacts', 'Apps installed on devices', 'Cellular network names', 'Dates of birth',
     'Device information', 'Email addresses', 'Genders', 'Geographic locations', 'IMEI numbers', 'IMSI numbers',
     'IP addresses', 'Names', 'Phone numbers', 'Profile photos', 'Social media profiles'],

    ['Email addresses', 'IP addresses', 'Passwords', 'Private messages', 'Usernames', 'Website activity'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Education levels', 'Email addresses', 'Genders', 'Geographic locations', 'Job applications',
     'Marital statuses', 'Names', 'Nationalities', 'Passwords', 'Phone numbers', 'Profile photos'],

    ['Email addresses', 'Email messages'],
    ['Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Homepage URLs', 'Instant messenger identities', 'IP addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Passwords', 'Physical addresses',
     'Usernames'],

    ['Email addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Geographic locations', 'Names', 'Passwords'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Employers', 'Geographic locations', 'Job titles', 'Names', 'Phone numbers', 'Salutations',
     'Social media profiles'],

    ['Ages', 'Auth tokens', 'Email addresses', 'Employment statuses', 'Genders', 'IP addresses', 'Marital statuses',
     'Names', 'Passwords', 'Physical addresses', 'Private messages', 'Social media profiles'],

    ['Email addresses', 'Employers', 'IP addresses', 'Names', 'Passwords', 'Phone numbers'],
    ['Browser user agent details', 'Email addresses', 'IP addresses', 'Names', 'Passwords'],
    ['Bios', 'Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Passwords',
     'Usernames'],

    ['Avatars', 'Email addresses', 'Geographic locations', 'IP addresses', 'Names', 'Passwords', 'Usernames',
     'Website activity'],

    ['Email addresses', 'IP addresses', 'Names', 'Passwords'],
    ['Email addresses', 'Names', 'Passwords', 'Salutations', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Ethnicities', 'Genders', 'Names', 'Passwords', 'Payment histories',
     'Phone numbers', 'Physical addresses', 'Security questions and answers', 'Sexual orientations', 'Usernames',
     'Website activity'],

    ['Email addresses', 'Instant messenger identities', 'IP addresses', 'Names', 'Passwords', 'Private messages',
     'Usernames', 'Website activity'],

    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Account balances', 'Email addresses', 'Names', 'Phone numbers'],
    ['Dates of birth', 'Drivers licenses', 'Email addresses', 'Names', 'Phone numbers', 'Physical addresses',
     'Social security numbers', 'Vehicle details'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Phone numbers', 'Physical addresses', 'Vehicle details',
     'Vehicle identification numbers (VINs)'],

    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Employers', 'Job titles', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Account balances', 'Email addresses', 'Genders', 'Government issued IDs', 'Names', 'Phone numbers',
     'Physical addresses'],

    ['Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Beauty ratings', 'Car ownership statuses', 'Dates of birth', 'Drinking habits', 'Education levels',
     'Email addresses', 'Genders', 'Geographic locations', 'Home ownership statuses', 'Income levels', 'IP addresses',
     'Job titles', 'Names', 'Passwords', 'Personal descriptions', 'Personal interests', 'Physical attributes',
     'Sexual orientations', 'Smoking habits', 'Website activity'],

    ['Credit cards', 'Genders', 'Passwords', 'Usernames'],
    ['Email addresses', 'Geographic locations', 'IP addresses', 'Job titles', 'Names', 'Passwords', 'Phone numbers',
     'Spoken languages', 'Survey results', 'Usernames'],

    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Passwords',
     'Private messages', 'Usernames'],

    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Career levels', 'Education levels', 'Email addresses', 'Names', 'Passwords', 'Phone numbers',
     'Physical addresses', 'Salutations', 'User website URLs', 'Website activity'],

    ['Ages', 'Email addresses', 'Genders', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Private messages', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Passwords', 'Security questions and answers',
     'Usernames', 'Website activity'],

    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Instant messenger identities', 'IP addresses', 'Passwords', 'Usernames',
     'Website activity'],

    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Device information', 'Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Passwords',
     'Usernames'],

    ['Browser user agent details', 'Email addresses', 'IP addresses', 'Passwords', 'Purchases', 'Usernames',
     'Website activity'],

    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Historical passwords', 'IP addresses', 'Names', 'Partial credit card data', 'Passwords',
     'Phone numbers', 'Physical addresses', 'Purchases'],

    ['Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Passwords', 'Phone numbers',
     'Social media profiles'],

    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Dates of birth', 'Email addresses', 'Flights taken', 'IP addresses', 'Names', 'Phone numbers',
     'Physical addresses', 'Purchases'],

    ['Dates of birth', 'Email addresses', 'Geographic locations', 'Historical passwords',
     'Instant messenger identities', 'IP addresses', 'Passwords', 'Private messages', 'User website URLs', 'Usernames'],

    ['Email addresses', 'Genders', 'IP addresses', 'Passwords', 'Private messages', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Account balances', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Dates of birth', 'Email addresses', 'Financial transactions', 'Geographic locations', 'IP addresses', 'Names',
     'Usernames'],

    ['Email addresses', 'IP addresses', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Names', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Dates of birth', 'Email addresses', 'Geographic locations', 'Historical passwords',
     'Instant messenger identities', 'IP addresses', 'Passwords', 'Private messages', 'Usernames', 'Website activity'],

    ['Email addresses', 'Geographic locations', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'Employers', 'Job titles', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords', 'Physical addresses'],
    ['Email addresses', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Phone numbers'],
    ['Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Social media profiles'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Government issued IDs', 'Names', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Passwords', 'Phone numbers',
     'Physical addresses'],

    ['Account balances', 'Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Passwords',
     'Payment histories', 'Payment methods', 'Physical addresses', 'Usernames', 'Website activity'],

    ['Email addresses', 'Family members names', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses'],
    ['Email addresses', 'Partial phone numbers'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Passwords', 'Physical addresses'],
    ['Biometric data', 'Dates of birth', 'Email addresses', 'Family members names', 'Genders', 'Job titles',
     'Marital statuses', 'Names', 'Passport numbers', 'Phone numbers', 'Physical addresses', 'Physical attributes'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Job titles', 'Names', 'Phone numbers', 'Physical addresses', 'Social media profiles'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Private messages', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Personal health data', 'Phone numbers',
     'Physical addresses', 'Salutations', 'Usernames'],

    ['Dates of birth', 'Drinking habits', 'Email addresses', 'Family structure', 'Genders', 'Geographic locations',
     'HIV statuses', 'IP addresses', 'Names', 'Passwords', 'Personal health data', 'Phone numbers',
     'Physical attributes', 'Private messages', 'Profile photos', 'Religions', 'Sexual orientations', 'Smoking habits',
     'Usernames'],

    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Email addresses'],
    ['Email addresses', 'IP addresses', 'Passwords'],
    ['Email addresses', 'Employers', 'IP addresses', 'Job titles', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Employers', 'Geographic locations', 'Job titles', 'Names', 'Phone numbers',
     'Social media profiles'],

    ['Buying preferences', 'Charitable donations', 'Credit status information', 'Dates of birth', 'Email addresses',
     'Family structure', 'Financial investments', 'Home ownership statuses', 'Income levels', 'Job titles',
     'Marital statuses', 'Names', 'Net worths', 'Phone numbers', 'Physical addresses', 'Political donations'],

    ['Email addresses', 'Geographic locations', 'IP addresses', 'Names', 'Passwords'],
    ['Email addresses', 'Passwords', 'Security questions and answers', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses',
     'Social security numbers'],

    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Names',
     'Spoken languages', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Partial credit card data', 'Passwords', 'Purchases'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Passwords', 'Private messages'],
    ['Dates of birth', 'Eating habits', 'Email addresses', 'IP addresses', 'Names', 'Passwords',
     'Physical attributes', 'Usernames'],
    ['Email addresses', 'Email messages', 'IP addresses', 'Names'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Names', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Names', 'Phone numbers', 'Physical addresses', 'Purchases'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Geographic locations', 'Names', 'Partial credit card data'],
    ['Browser user agent details', 'Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses', 'Vehicle details'],
    ['Dates of birth', 'Device information', 'Email addresses', 'IP addresses', 'Names', 'Passwords', 'Phone numbers',
     'Physical addresses'],

    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Geographic locations', 'Names', 'Passwords', 'Phone numbers', 'Spoken languages', 'Usernames'],

    ['Dates of birth', 'Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Device information', 'Email addresses', 'Geographic locations', 'IP addresses', 'Names', 'Phone numbers'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers', 'Social media profiles'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Partial credit card data', 'Passwords', 'Phone numbers',
     'Physical addresses', 'Social media profiles'],

    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Employers', 'Geographic locations', 'Passwords', 'Phone numbers', 'Usernames'],
    ['Email addresses', 'Geographic locations', 'Usernames'],
    ['Email addresses', 'Employers', 'Names', 'Physical addresses'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Purchases', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Phone numbers', 'Physical addresses', 'Purchases'],
    ['Email addresses', 'IP addresses', 'Names', 'Passwords', 'Payment histories', 'Phone numbers',
     'Physical addresses', 'Usernames', 'Website activity'],

    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Deceased date', 'Email addresses', 'Genders', 'Government issued IDs', 'Names',
     'Passport numbers', 'Passwords', 'Phone numbers', 'Physical addresses', 'Utility bills'],

    ['Email addresses', 'IP addresses', 'Passwords', 'Private messages', 'Usernames', 'Website activity'],
    ['Dates of birth', 'Email addresses', 'Geographic locations', 'Job applications', 'Names', 'Passwords',
     'Phone numbers', 'Spoken languages'],

    ['Dates of birth', 'Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Credit status information', 'Dates of birth', 'Education levels', 'Email addresses', 'Ethnicities',
     'Family structure', 'Financial investments', 'Genders', 'Home ownership statuses', 'Income levels', 'IP addresses',
     'Marital statuses', 'Names', 'Net worths', 'Occupations', 'Personal interests', 'Phone numbers',
     'Physical addresses', 'Religions', 'Spoken languages'],

    ['Credit status information', 'Dates of birth', 'Email addresses', 'Ethnicities', 'Family structure', 'Genders',
     'Home ownership statuses', 'Income levels', 'IP addresses', 'Names', 'Phone numbers', 'Physical addresses',
     'Purchasing habits'],

    ['Email addresses', 'Employers', 'Government issued IDs', 'Names', 'Occupations', 'Phone numbers'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Family structure', 'Genders', 'Names', 'Phone numbers', 'Physical addresses',
     'Vehicle details'],
    ['Bios', 'Email addresses', 'Names', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Employers', 'Genders', 'Geographic locations', 'Names', 'Phone numbers',
     'Relationship statuses'],

    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers', 'Usernames'],
    ['Email addresses', 'Employers', 'Phone numbers', 'Physical addresses'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords'],
    ['Email addresses', 'Genders', 'Names', 'Partial dates of birth', 'Passwords', 'Phone numbers',
     'Physical addresses', 'Purchases', 'Social media profiles']
    ,
    ['Email addresses', 'IP addresses', 'Names', 'Passwords', 'Purchases', 'Usernames'],
    ['Browser user agent details', 'Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Passwords',
     'Phone numbers', 'Physical addresses', 'Purchases'],

    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Government issued IDs', 'Physical addresses'],
    ['Email addresses', 'Names', 'Partial credit card data', 'Passwords', 'Phone numbers'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Passwords', 'Phone numbers', 'Sexual fetishes', 'Sexual orientations', 'Usernames', 'Website activity'],


    ['Dates of birth', 'Email addresses', 'Names', 'Passwords', 'School grades (class levels)', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Passwords', 'User website URLs', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Account balances', 'Browser user agent details', 'Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses', 'Usernames', 'Website activity'],


    ['Email addresses', 'Passwords', 'Usernames'],
    ['Device information', 'Email addresses', 'Names', 'Phone numbers', 'Physical addresses', 'Purchases'],
    ['Email addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Dates of birth', 'Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Names', 'Passwords', 'Phone numbers','Usernames'],


    ['Avatars', 'Email addresses', 'Names', 'Passwords', 'Private messages', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Encrypted keys', 'Mnemonic phrases', 'Passwords'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords', 'Social media profiles'],
    ['Email addresses', 'Geographic locations', 'Names', 'Professional skills', 'Usernames',  'Years of professional experience'],


    ['Email addresses', 'Partial phone numbers'],
    ['Email addresses', 'Genders', 'IP addresses', 'Names', 'Passwords', 'Social media profiles'],
    ['Email addresses', 'IP addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses', 'Purchases'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Geographic locations', 'Names', 'Purchases'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Bank account numbers', 'Email addresses', 'IP addresses', 'Names', 'Partial credit card data', 'Passport numbers',
     'Phone numbers', 'Physical addresses', 'Purchases', 'Security questions and answers', 'Social security numbers'],

    ['Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Passwords', 'Physical addresses', 'Security questions and answers', 'Usernames', 'Website activity'],


    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Names', 'Partial credit card data', 'Passwords', 'Phone numbers','Physical addresses', 'Purchases'],


    ['Browser user agent details', 'Email addresses', 'Geographic locations', 'IP addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses', 'Salutations'],


    ['Dates of birth', 'Email addresses', 'Instant messenger identities', 'IP addresses', 'Passwords', 'Social connections', 'Spoken languages', 'Time zones', 'User website URLs', 'Usernames', 'Website activity'],


    ['Email addresses', 'Email messages'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'Names', 'Passwords'],
    ['Email addresses', 'Geographic locations', 'Names', 'Passwords', 'Phone numbers'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Health insurance information', 'IP addresses', 'Names', 'Personal health data', 'Phone numbers', 'Physical addresses', 'Security questions and answers',   'Social connections'],



    ['Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Genders', 'Nicknames', 'Partial dates of birth', 'Passwords', 'Usernames'],
    ['Browser user agent details', 'Email addresses', 'IP addresses', 'Usernames', 'Website activity'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Geographic locations', 'IP addresses', 'Names', 'Partial credit card data', 'Passwords', 'Phone numbers'],


    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Website activity'],
    ['Email addresses', 'Geographic locations', 'IP addresses', 'Names', 'Passwords', 'Social media profiles', 'Usernames'],


    ['Dates of birth', 'Email addresses', 'Historical passwords', 'IP addresses', 'Passwords', 'Usernames'],
    ['Browser user agent details', 'Dates of birth', 'Email addresses', 'IP addresses', 'Names', 'Phone numbers',  'Physical addresses', 'Purchases', 'Usernames'],


    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Names', 'Passwords', 'Phone numbers', 'Social media profiles'],


    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Geographic locations', 'IP addresses', 'Job applications', 'Job titles',  'Names', 'Passwords', 'Phone numbers'],


    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Genders', 'Names', 'Partial credit card data', 'Passwords', 'Phone numbers', 'Physical addresses', 'Purchases', 'Usernames'],


    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Geographic locations', 'Names', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Passwords', 'Usernames'],
    ['Auth tokens', 'Device information', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers','Salutations', 'Social media profiles', 'Usernames'],


    ['Email addresses', 'Geographic locations', 'Passwords'],
    ['Email addresses', 'Passwords', 'Phone numbers', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Names', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'Government issued IDs',  'Marital statuses', 'Names', 'Nationalities', 'Passwords', 'Phone numbers', 'Physical addresses', 'Usernames'],


    ['Email addresses', 'Names', 'Passwords', 'Payment histories', 'Usernames'],
    ['Email addresses', 'Employers', 'IP addresses', 'Names', 'Occupations', 'Passwords', 'Phone numbers'],
    ['Dates of birth', 'Email addresses', 'Geographic locations', 'Names'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Avatars', 'Dates of birth', 'Email addresses', 'IP addresses', 'Website activity'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'Recovery email addresses', 'Security questions and answers', 'Usernames'],


    ['Email addresses', 'Email messages', 'Genders', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Geographic locations', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Dates of birth', 'Email addresses', 'Employers', 'Family structure', 'Genders', 'Income levels', 'Living costs', 'Marital statuses', 'Mothers maiden names', 'Names', 'Phone numbers', 'Physical addresses', 'Places of birth', 'Religions', 'Spouses names'],



    ['Bank account numbers', 'Dates of birth', 'Email addresses', 'Genders', 'Names', 'Phone numbers', 'Physical addresses'],


    ['Email addresses', 'Names', 'Passwords', 'Physical addresses', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Names', 'Partial credit card data', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Genders', 'IP addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Geographic locations', 'IP addresses', 'Passwords', 'Private messages',  'Usernames'],


    ['Auth tokens', 'Dates of birth', 'Education levels', 'Email addresses', 'Geographic locations', 'IP addresses', 'Names', 'Passwords', 'Phone numbers', 'Private messages', 'Security questions and answers','Social media profiles', 'Usernames'],



    ['Email addresses', 'Passwords'],
    ['Education levels', 'Email addresses', 'Genders', 'Geographic locations', 'Job titles', 'Names', 'Social media profiles'],


    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Avatars', 'Dates of birth', 'Email addresses', 'Geographic locations', 'IP addresses', 'Passwords', 'Time zones',   'Website activity'],


    ['Dates of birth', 'Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Phone numbers', 'Physical addresses', 'Purchases', 'Salutations'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Names', 'Passwords', 'Usernames', 'Website activity'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'IP addresses', 'Names', 'Passwords', 'Private messages', 'Usernames', 'Website activity'],
    ['Auth tokens', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Spoken languages', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords', 'Phone numbers'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Auth tokens', 'Avatars', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Social media profiles',   'Usernames'],


    ['Email addresses', 'Geographic locations', 'Passwords', 'Usernames'],
    ['Auth tokens', 'Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Names', 'Partial dates of birth', 'Social media profiles'],


    ['Dates of birth', 'Deceased statuses', 'Email addresses', 'Employers', 'Ethnicities', 'Genders', 'Government issued IDs', 'Home ownership statuses', 'Job titles', 'Names', 'Nationalities', 'Phone numbers',  'Physical addresses'],



    ['Email addresses', 'IP addresses', 'Names', 'Partial credit card data', 'Phone numbers', 'Salutations'],
    ['Astrological signs', 'Dates of birth', 'Drinking habits', 'Drug habits', 'Education levels', 'Email addresses', 'Ethnicities', 'Fitness levels', 'Genders', 'Geographic locations', 'Income levels', 'Job titles', 'Names',  'Parenting plans', 'Passwords', 'Personal descriptions', 'Physical attributes', 'Political views','Relationship statuses', 'Religions', 'Sexual fetishes', 'Travel habits', 'Usernames', 'Website activity',  'Work habits'],





    ['Device information', 'Email addresses', 'Names', 'Passwords', 'Social media profiles'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Email messages', 'IP addresses', 'Names'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Drinking habits', 'Drug habits', 'Email addresses', 'Genders', 'Geographic locations','IP addresses', 'Marital statuses', 'Names     'Sexual orientations', 'Smoking habits', 'Social media profiles', 'Usernames'],



    ['Dates of birth', 'Email addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Dates of birth', 'Email addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Dates of birth', 'Email addresses', 'Names'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Appointments', 'Dates of birth', 'Email addresses', 'Genders', 'Marital statuses', 'Names', 'Passwords',  'Phone numbers', 'Physical addresses'],


    ['Dates of birth', 'Email addresses', 'Genders', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Job titles', 'Names', 'Phone numbers', 'Physical addresses'],


    ['Dates of birth', 'Email addresses', 'IP addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Genders', 'Geographic locations', 'Passwords', 'Phone numbers'],
    ['Email addresses', 'Names', 'Passwords', 'Phone numbers'],
    ['Email addresses', 'Names', 'Passwords', 'Physical addresses', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Social connections', 'Usernames',    'Website activity'],


    ['Device usage tracking data'],
    ['Age groups', 'Email addresses', 'Employers', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses','Website activity'],


    ['Chat logs', 'Email addresses', 'Geographic locations', 'IP addresses', 'Passwords', 'Private messages','User statuses', 'Usernames'],


    ['Credit status information', 'Email addresses', 'Home loan information', 'Income levels', 'IP addresses', 'Names','Passwords', 'Personal descriptions', 'Physical addresses'],


    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses', 'Purchases'],


    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Names'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Names', 'Passwords',     'Usernames'],


    ['Email addresses', 'Passwords'],
    ['Account balances', 'Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Phone numbers','Physical addresses', 'Security questions and answers', 'Website activity'],


    ['Dates of birth', 'Email addresses', 'IP addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses', 'Usernames'],


    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Employers', 'Job titles', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Dates of birth', 'Email addresses', 'Names', 'Purchases'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords'],
    ['Avatars', 'Dates of birth', 'Email addresses', 'Genders', 'Names', 'Spoken languages', 'Usernames',  'Website activity'],


    ['Email addresses', 'IP addresses', 'Names', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'IP addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Private messages', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Private messages', 'Usernames',  'Website activity'],


    ['Email addresses'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Private messages', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Private messages', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Avatars', 'Email addresses', 'IP addresses', 'Phone numbers', 'Physical addresses', 'Purchases','Social media profiles', 'Usernames'],


    ['Email addresses', 'Geographic locations', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Names', 'Passwords', 'Phone numbers'],
    ['Email addresses', 'Email messages', 'Geographic locations', 'Passwords', 'Phone numbers'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Bank account numbers', 'Dates of birth', 'Email addresses', 'Genders', 'Names', 'Partial credit card data','Payment histories', 'Phone numbers', 'Physical addresses'],

    ['Account balances', 'Dates of birth', 'Email addresses', 'IP addresses', 'Names', 'Phone numbers','Physical addresses', 'Security questions and answers', 'Usernames', 'Website activity'],


    ['Browser user agent details', 'Email addresses', 'IP addresses', 'Usernames'],
    ['Email addresses', 'Licence plates', 'Names', 'Passwords', 'Phone numbers'],
    ['Email addresses', 'Passwords', 'Payment histories', 'Physical addresses', 'Private messages', 'Website activity'],

    ['Browser user agent details', 'Email addresses', 'IP addresses', 'Names', 'Partial credit card data', 'Passwords',   'Phone numbers', 'Website activity'],


    ['Email addresses', 'IP addresses', 'Names', 'Partial credit card data', 'Phone numbers', 'Physical addresses',  'Purchases'],


    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'Income levels', 'Names', 'Phone numbers', 'Purchases'],


    ['Email addresses', 'Names', 'Passwords'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Nationalities', 'Phone numbers', 'Physical addresses'],

    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Geographic locations', 'Names', 'Passwords', 'Social media profiles'],
    ['Email addresses', 'Genders', 'Geographic locations', 'Names', 'Passwords', 'Social media profiles', 'Usernames','Website activity'],


    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Passwords', 'Phone numbers','Physical addresses', 'Purchases'],


    ['Email addresses', 'Job titles', 'Names', 'Passwords', 'Phone numbers', 'Social media profiles'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Device information', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Passwords','Social media profiles', 'Usernames'],


    ['Email addresses', 'IP addresses', 'Passwords', 'Time zones', 'Usernames', 'Website activity'],
    ['Email addresses', 'Genders', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'IP addresses', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Genders', 'Geographic locations', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Social media profiles', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Genders', 'IP addresses', 'Names', 'Passwords'],
    ['Email addresses', 'Names'],
    ['Dates of birth', 'Device information', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Passwords'],
    ['Email addresses', 'Genders', 'Job applications', 'Marital statuses', 'Names', 'Nationalities', 'Passport numbers','Passwords', 'Phone numbers', 'Physical addresses', 'Religions', 'Salutations'],


    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Bank account numbers', 'Customer feedback', 'Dates of birth', 'Financial transactions', 'Genders','Geographic locations', 'Government issued IDs', 'IP addresses', 'Marital statuses', 'Names', 'Passwords','Phone numbers', 'Physical addresses', 'PINs', 'Security questions and answers', 'Spoken languages'],


    ['Email addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Private messages', 'Usernames', 'Website activity'],
    ['Browser user agent details', 'Email addresses', 'IP addresses', 'Survey results'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Employers', 'Job titles', 'Names', 'Phone numbers'],
    ['Browser user agent details', 'Email addresses', 'IP addresses', 'Names', 'Passwords'],
    ['Email addresses', 'Names', 'Passwords'],
    ['Email addresses', 'Genders', 'Passwords', 'Phone numbers', 'Usernames'],
    ['Email addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Occupations', 'Passwords', 'Phone numbers'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Browser user agent details', 'Credit card CVV', 'Email addresses', 'IP addresses', 'Names', 'Partial credit card data', 'Phone numbers', 'Physical addresses', 'Purchases'],


    ['Email addresses', 'Names', 'Passwords'],
    ['Browser user agent details', 'Email addresses', 'Geographic locations', 'IP addresses', 'Names'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Names', 'Physical addresses'],
    ['Email addresses'],
    ['Email addresses', 'IP addresses', 'Names', 'Partial credit card data', 'Passwords'],
    ['Geographic locations', 'IP addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses','Social media profiles', 'Vehicle details'],


    ['Email addresses', 'Names', 'Passwords', 'Phone numbers'],
    ['Email addresses', 'Genders', 'Names', 'Physical addresses'],
    ['Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Physical addresses'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Password strengths', 'Passwords'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Ethnicities', 'Genders', 'Names', 'Physical attributes'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Names', 'Passwords'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Names', 'Physical addresses', 'Private messages', 'Purchases'],
    ['Email addresses', 'Genders', 'IP addresses', 'Passwords', 'Private messages', 'Usernames'],
    ['Email addresses', 'Geographic locations', 'Names', 'Passwords', 'Phone numbers'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses','Social media profiles', 'Usernames'],


    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Bios', 'Email addresses', 'IP addresses', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Phone numbers', 'Physical addresses', 'Purchases'],
    ['Email addresses', 'Names', 'Passwords'],
    ['Email addresses', 'Genders', 'Geographic locations', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Geographic locations', 'Phone numbers', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Private messages', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Names', 'Passwords', 'Phone numbers'],
    ['Email addresses', 'Names', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses', 'Usernames'],


    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'Names', 'Passwords','Social connections'],


    ['Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Physical addresses'],
    ['Email addresses', 'Employers', 'Job titles', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Audio recordings', 'Browsing histories', 'Device information', 'Email addresses', 'Geographic locations','IMEI numbers', 'IP addresses', 'Names', 'Passwords', 'Photos', 'SMS messages'],


    ['Credit cards', 'Email addresses', 'IP addresses', 'Passwords', 'Support tickets', 'Usernames'],
    ['Customer interactions', 'Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'MAC addresses', 'Names', 'Passport numbers', 'Passwords', 'Phone numbers'],


    ['Email addresses', 'Geographic locations', 'Names', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Physical addresses', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses',  'Spoken languages'],


    ['Email addresses', 'Names', 'Passwords', 'Physical addresses', 'Purchases', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'Genders', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Credit cards', 'Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Device information', 'Device serial numbers', 'Email addresses', 'Geographic locations', 'IMSI numbers', 'Login histories'],


    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Partial credit card data', 'Passwords', 'Phone numbers', 'Profile photos'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'Names', 'Passwords', 'Phone numbers','Usernames'],


    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Names', 'Passwords'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Nationalities', 'Phone numbers', 'Physical addresses', 'Salutations', 'Spoken languages'],


    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Chat logs', 'Email addresses', 'IP addresses', 'Passwords', 'Private messages', 'Time zones'],
    ['Email addresses', 'Geographic locations', 'Names', 'Social media profiles'],
    ['Email addresses', 'Email messages'],
    ['Avatars', 'Email addresses', 'IP addresses', 'Passwords', 'Payment histories', 'Private messages', 'Usernames', 'Website activity'],


    ['Email addresses', 'Passwords', 'Reward program balances'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Geographic locations', 'IP addresses', 'Passwords', 'Usernames',  'Website activity'],


    ['Email addresses', 'Passwords', 'Usernames'],
    ['Age groups', 'Credit cards', 'Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers','Physical addresses', 'Purchases', 'Usernames'],


    ['Email addresses', 'IP addresses', 'Names', 'Phone numbers', 'Physical addresses', 'Purchases'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Names', 'Passwords', 'Physical addresses', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Usernames'],
    ['Bank account numbers', 'Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Payment histories', 'Phone numbers', 'Physical addresses'],


    ['Email addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Passwords'],
    ['Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Bios', 'Email addresses', 'Geographic locations', 'IP addresses', 'Names', 'Passwords', 'Usernames'],
    ['Age groups', 'Dates of birth', 'Email addresses', 'Genders', 'Names', 'Physical addresses'],
    ['Auth tokens', 'Dates of birth', 'Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],

    ['Email addresses'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Account balances', 'Dates of birth', 'Email addresses', 'Names', 'Passwords', 'Phone numbers',   'Physical addresses', 'Usernames'],


    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Bios', 'Email addresses', 'Geographic locations', 'Names', 'Phone numbers', 'Profile photos', 'Usernames'],
    ['Email addresses', 'Names', 'Social media profiles', 'Usernames'],
    ['Email addresses', 'Genders', 'Names', 'Social connections', 'Website activity'],
    ['Bios', 'Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Social media profiles'],
    ['Email addresses', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Education levels', 'Email addresses', 'Ethnicities', 'Genders', 'Job titles', 'Names','Phone numbers', 'Physical addresses', 'Social security numbers'],


    ['Email addresses', 'Passwords', 'Usernames'],
    ['Bank account numbers', 'Credit status information', 'Dates of birth', 'Email addresses', 'Employers','Health insurance information', 'Income levels', 'IP addresses', 'Names', 'Personal health data', 'Phone numbers','Physical addresses', 'Smoking habits', 'Social security numbers'],



    ['Bank account numbers', 'Dates of birth', 'Email addresses', 'Family members names', 'Genders','Government issued IDs', 'Income levels', 'Marital statuses', 'Nationalities', 'Occupations', 'Passwords', 'Phone numbers', 'Physical addresses'],


    ['Email addresses', 'Genders', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Names', 'Passwords', 'Phone numbers'],
    ['Email addresses', 'Names', 'Personal health data', 'Social security numbers'],
    ['Dates of birth', 'Email addresses', 'Homepage URLs', 'Instant messenger identities', 'IP addresses', 'Passwords',     'Security questions and answers', 'Spoken languages', 'Website activity'],


    ['Browser user agent details', 'Email addresses', 'Genders', 'IP addresses', 'Names', 'Passwords', 'Phone numbers',  'Spoken languages', 'Time zones', 'Website activity'],


    ['Dates of birth', 'Email addresses', 'Employers', 'Genders', 'Geographic locations', 'IP addresses', 'Job titles','Names', 'Phone numbers', 'Physical addresses'],


    ['Email addresses', 'Historical passwords', 'IP addresses', 'Passwords', 'Private messages', 'Usernames',     'Website activity'],


    ['Email addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'Government issued IDs', 'Names', 'Passwords', 'Phone numbers'],
    ['Email addresses', 'Names', 'Passwords', 'Phone numbers'],
    ['Dates of birth', 'Email addresses', 'Genders', 'IP addresses', 'Marital statuses', 'Names', 'Occupations', 'Passwords', 'Phone numbers', 'Physical addresses', 'Usernames'],


    ['Credit cards', 'Email addresses', 'Government issued IDs', 'IP addresses', 'Names', 'Passwords', 'Phone numbers','Physical addresses', 'Purchases', 'SMS messages', 'Usernames'],


    ['Email addresses', 'IP addresses', 'Passwords', 'Private messages', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Family members names', 'Genders', 'IP addresses', 'Names', 'Passwords',  'Physical addresses', 'Security questions and answers', 'Usernames', 'Website activity'],

    ['Email addresses', 'IP addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Browser user agent details', 'Email addresses', 'IP addresses', 'Names', 'Physical addresses', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Names', 'Passwords', 'Physical addresses'],
    ['Email addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'Usernames', 'Website activity'],
    ['Dates of birth', 'Email addresses', 'Passwords', 'Usernames'],
    ['Bios', 'Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Names', 'Passwords', 'Social media profiles', 'User website URLs', 'Usernames'],


    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Genders', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Browser user agent details', 'Email addresses', 'Employers', 'IP addresses', 'Names', 'Partial credit card data','Physical addresses', 'Purchases'],


    ['Education levels', 'Email addresses', 'IP addresses', 'Job applications', 'Names', 'Passwords', 'Phone numbers',  'Physical addresses'],


    ['Email addresses', 'Names', 'Passwords'],
    ['Email addresses', 'Email messages', 'Employers', 'IP addresses', 'Names', 'Partial credit card data', 'Passwords', 'Payment histories', 'Physical addresses', 'Website activity'],


    ['Dates of birth', 'Email addresses', 'Names', 'Phone numbers', 'Physical addresses'],
    ['Email addresses', 'IP addresses', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Instant messenger identities', 'IP addresses', 'Names', 'Passwords', 'Private messages','Usernames', 'Website activity'],


    ['Auth tokens', 'Dates of birth', 'Email addresses', 'Genders', 'Names', 'Phone numbers', 'Usernames'],
    ['Auth tokens', 'Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Names', 'Passwords', 'Phone numbers', 'Profile photos', 'Social media profiles', 'Usernames'],


    ['Dates of birth', 'Email addresses', 'IP addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses'],
    ['Dates of birth', 'Email addresses', 'Geographic locations', 'IP addresses', 'Names', 'Passwords', 'Phone numbers', 'Social media profiles'],


    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames', 'Website activity'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Names', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses', 'Usernames'],

    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Email addresses', 'Names', 'Passwords', 'Phone numbers', 'Physical addresses', 'PINs'],
    ['Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'IP addresses', 'Names', 'Social media profiles', 'Usernames'],
    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Employers', 'Geographic locations', 'Job titles', 'Names', 'Social media profiles'],
    ['Browser user agent details', 'Chat logs', 'Email addresses', 'IP addresses', 'Names', 'Phone numbers','Physical addresses', 'Purchases'],

    ['Email addresses', 'Passwords'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Email addresses', 'IP addresses', 'Names', 'Passwords', 'Phone numbers'],
    ['Email addresses', 'Passwords'],
    ['Dates of birth', 'Drinking habits', 'Education levels', 'Email addresses', 'Ethnicities', 'Family structure', 'Genders', 'Geographic locations', 'Income levels', 'Names', 'Nicknames', 'Physical attributes', 'Political views','Relationship statuses', 'Religions', 'Sexual orientations', 'Smoking habits'],


    ['Dates of birth', 'Email addresses', 'IP addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Names', 'Vehicle details'],
    ['Email addresses', 'Passwords', 'Phone numbers', 'Usernames'],
    ['Email addresses', 'Passwords', 'Usernames'],
    ['Dates of birth', 'Email addresses', 'Genders', 'Geographic locations', 'IP addresses', 'Passwords', 'Spoken languages'],


    ['Email addresses', 'Names', 'Phone numbers', 'Usernames'],





]

# Finding the frequent patterns with min support threshold=0.5
FrequentPatterns = pyfpgrowth.find_frequent_patterns(transactions=transactions, support_threshold=0.5)
#print(FrequentPatterns)

# Generating rules with min confidence threshold=0.5
Rules = pyfpgrowth.generate_association_rules(patterns=FrequentPatterns, confidence_threshold=0.5)
print(Rules)

with open('c:/temp/xxx.txt', 'w') as f:
        f.write(str(Rules))
print("DONE")