import pandas as pd
import math
import numpy as np


def WebDomain2Country(row):

    sDomain=str(row['Domain'])


    s='not found'

    if sDomain[-3:]=='.ae' : s='abu dhabi'
    if sDomain[-3:]=='.af': s='afghanistan'
    if sDomain[-3:]=='.ax': s='åland'
    if sDomain[-3:]=='.al': s='albania'
    if sDomain[-3:]=='.dz': s='algeria'
    if sDomain[-3:]=='.as': s='american samoa'
    if sDomain[-3:]=='.ad': s='andorra'
    if sDomain[-3:]=='.ao': s='angola'
    if sDomain[-3:]=='.ai': s='anguilla'
    if sDomain[-3:]=='.aq': s='antarctica'
    if sDomain[-3:]=='.ag': s='antigua and barbuda'
    if sDomain[-3:]=='.ar': s='argentina'
    if sDomain[-3:]=='.am': s='armenia'
    if sDomain[-3:]=='.aw': s='aruba'
    if sDomain[-3:]=='.ac': s='ascension island'
    if sDomain[-3:]=='.au': s='australia'
    if sDomain[-3:]=='.at': s='austria'
    if sDomain[-3:]=='.az': s='azerbaijan'
    if sDomain[-3:]=='.bs': s='bahamas'
    if sDomain[-3:]=='.bh': s='bahrain'
    if sDomain[-3:]=='.bd': s='bangladesh'
    if sDomain[-3:]=='.bb': s='barbados'
    if sDomain[-3:]=='eus': s='basque country'
    if sDomain[-3:]=='.by': s='belarus'
    if sDomain[-3:]=='.be': s='belgium'
    if sDomain[-3:]=='.bz': s='belize'
    if sDomain[-3:]=='.bj': s='benin'
    if sDomain[-3:]=='.bm': s='bermuda'
    if sDomain[-3:]=='.bt': s='bhutan'
    if sDomain[-3:]=='.bo': s='bolivia'
    if sDomain[-3:]=='.bq': s='bonaire'
    if sDomain[-3:]=='.ba': s='bosnia and herzegovina'
    if sDomain[-3:]=='.bw': s='botswana'
    if sDomain[-3:]=='.bv': s='bouvet island'
    if sDomain[-3:]=='.br': s='brazil'
    if sDomain[-3:]=='.io': s='british indian ocean territory'
    if sDomain[-3:]=='.vg': s='british virgin islands'
    if sDomain[-3:]=='.bn': s='brunei'
    if sDomain[-3:]=='.bg': s='bulgaria'
    if sDomain[-3:]=='.bf': s='burkina faso'
    if sDomain[-3:]=='.mm': s='burma (officially: myanmar)'
    if sDomain[-3:]=='.bi': s='burundi'
    if sDomain[-3:]=='.kh': s='cambodia'
    if sDomain[-3:]=='.cm': s='cameroon'
    if sDomain[-3:]=='.ca': s='canada'
    if sDomain[-3:]=='.cv': s='cape verde (in portuguese: cabo verde)'
    if sDomain[-3:]=='cat': s='catalonia'
    if sDomain[-3:]=='.ky': s='cayman islands'
    if sDomain[-3:]=='.cf': s='central african republic'
    if sDomain[-3:]=='.td': s='chad'
    if sDomain[-3:]=='.cl': s='chile'
    if sDomain[-3:]=='.cn': s='china, peoples republic of'
    if sDomain[-3:]=='.cx': s='christmas island'
    if sDomain[-3:]=='.cc': s='cocos (keeling) islands'
    if sDomain[-3:]=='.co': s='colombia'
    if sDomain[-3:]=='.km': s='comoros'
    if sDomain[-3:]=='.cd': s='congo, democratic republic of the (congo-kinshasa)'
    if sDomain[-3:]=='.cg': s='congo, republic of the (congo-brazzaville)'
    if sDomain[-3:]=='.ck': s='cook islands'
    if sDomain[-3:]=='.cr': s='costa rica'
    if sDomain[-3:]=='.ci': s='côte d’ivoire (ivory coast)'
    if sDomain[-3:]=='.hr': s='croatia'
    if sDomain[-3:]=='.cu': s='cuba'
    if sDomain[-3:]=='.cw': s='curaçao'
    if sDomain[-3:]=='.cy': s='cyprus'
    if sDomain[-3:]=='.nc': s='cyprus, north (unrecognised, self-declared state)'
    if sDomain[-3:]=='.cz': s='czechia (czech republic)'
    if sDomain[-3:]=='.dk': s='denmark'
    if sDomain[-3:]=='.dj': s='djibouti'
    if sDomain[-3:]=='.dm': s='dominica'
    if sDomain[-3:]=='.do': s='dominican republic'
    if sDomain[-3:]=='.ae': s='dubai'
    if sDomain[-3:]=='.tl': s='east timor (timor-leste)'
    if sDomain[-3:]=='.ec': s='ecuador'
    if sDomain[-3:]=='.eg': s='egypt'
    if sDomain[-3:]=='.sv': s='el salvador'
    if sDomain[-3:]=='.uk': s='england'
    if sDomain[-3:]=='.gq': s='equatorial guinea'
    if sDomain[-3:]=='.er': s='eritrea'
    if sDomain[-3:]=='.ee': s='estonia'
    if sDomain[-3:]=='.et': s='ethiopia'
    if sDomain[-3:]=='.eu': s='european union'
    if sDomain[-3:]=='.fo': s='faeroe islands'
    if sDomain[-3:]=='.fk': s='falkland islands'
    if sDomain[-3:]=='.fj': s='fiji'
    if sDomain[-3:]=='.fi': s='finland'
    if sDomain[-3:]=='.fr': s='france'
    if sDomain[-3:]=='.gf': s='french guiana (french overseas department)'
    if sDomain[-3:]=='.pf': s='french polynesia (french overseas collectivity)'
    if sDomain[-3:]=='.tf': s='french southern and antarctic lands'
    if sDomain[-3:]=='.ga': s='gabon (officially: gabonese republic)'
    if sDomain[-3:]=='.gal': s='galicia'
    if sDomain[-3:]=='.gm': s='gambia'
    if sDomain[-3:]=='.ps': s='gaza strip (gaza)'
    if sDomain[-3:]=='.ge': s='georgia'
    if sDomain[-3:]=='.de': s='germany'
    if sDomain[-3:]=='.gh': s='ghana'
    if sDomain[-3:]=='.gi': s='gibraltar'
    if sDomain[-3:]=='.uk': s='great britain (gb)'
    if sDomain[-3:]=='.gr': s='greece'
    if sDomain[-3:]=='.gl': s='greenland'
    if sDomain[-3:]=='.gd': s='grenada'
    if sDomain[-3:]=='.gp': s='guadeloupe (french overseas department)'
    if sDomain[-3:]=='.gu': s='guam'
    if sDomain[-3:]=='.gt': s='guatemala'
    if sDomain[-3:]=='.gg': s='guernsey'
    if sDomain[-3:]=='.gn': s='guinea'
    if sDomain[-3:]=='.gw': s='guinea-bissau'
    if sDomain[-3:]=='.gy': s='guyana'
    if sDomain[-3:]=='.ht': s='haiti'
    if sDomain[-3:]=='.hm': s='heard island and mcdonald islands'
    if sDomain[-3:]=='.nl': s='holland (officially: the netherlands)'
    if sDomain[-3:]=='.hn': s='honduras'
    if sDomain[-3:]=='.hk': s='hong kong'
    if sDomain[-3:]=='.hu': s='hungary'
    if sDomain[-3:]=='.is': s='iceland'
    if sDomain[-3:]=='.in': s='india'
    if sDomain[-3:]=='.id': s='indonesia'
    if sDomain[-3:]=='.ir': s='iran'
    if sDomain[-3:]=='.iq': s='iraq'
    if sDomain[-3:]=='.ie': s='ireland'
    if sDomain[-3:]=='.uk': s='ireland, northern'
    if sDomain[-3:]=='.im': s='isle of man'
    if sDomain[-3:]=='.il': s='israel'
    if sDomain[-3:]=='.it': s='italy'
    if sDomain[-3:]=='.jm': s='jamaica'
    if sDomain[-3:]=='.jp': s='japan'
    if sDomain[-3:]=='.je': s='jersey'
    if sDomain[-3:]=='.jo': s='jordan'
    if sDomain[-3:]=='.kz': s='kazakhstan'
    if sDomain[-3:]=='.ke': s='kenya'
    if sDomain[-3:]=='.ki': s='kiribati'
    if sDomain[-3:]=='.kp': s='korea, north'
    if sDomain[-3:]=='.kr': s='korea, south'
    if sDomain[-3:]=='al': s='kosovo'
    if sDomain[-3:]=='.kw': s='kuwait'
    if sDomain[-3:]=='.kg': s='kyrgyzstan'
    if sDomain[-3:]=='.la': s='laos'
    if sDomain[-3:]=='.lv': s='latvia'
    if sDomain[-3:]=='.lb': s='lebanon'
    if sDomain[-3:]=='.ls': s='lesotho'
    if sDomain[-3:]=='.lr': s='liberia'
    if sDomain[-3:]=='.ly': s='libya'
    if sDomain[-3:]=='.li': s='liechtenstein'
    if sDomain[-3:]=='.lt': s='lithuania'
    if sDomain[-3:]=='.lu': s='luxembourg'
    if sDomain[-3:]=='.mo': s='macau'
    if sDomain[-3:]=='.mk': s='macedonia, north'
    if sDomain[-3:]=='.mg': s='madagascar'
    if sDomain[-3:]=='.mw': s='malawi'
    if sDomain[-3:]=='.my': s='malaysia'
    if sDomain[-3:]=='.mv': s='maldives'
    if sDomain[-3:]=='.ml': s='mali'
    if sDomain[-3:]=='.mt': s='malta'
    if sDomain[-3:]=='.mh': s='marshall islands'
    if sDomain[-3:]=='.mq': s='martinique (french overseas department)'
    if sDomain[-3:]=='.mr': s='mauritania'
    if sDomain[-3:]=='.mu': s='mauritius'
    if sDomain[-3:]=='.yt': s='mayotte (french overseas department)'
    if sDomain[-3:]=='.mx': s='mexico'
    if sDomain[-3:]=='.fm': s='micronesia (officially: federated states of micronesia)'
    if sDomain[-3:]=='.md': s='moldova'
    if sDomain[-3:]=='.mc': s='monaco'
    if sDomain[-3:]=='.mn': s='mongolia'
    if sDomain[-3:]=='.me': s='montenegro'
    if sDomain[-3:]=='.ms': s='montserrat'
    if sDomain[-3:]=='.ma': s='morocco'
    if sDomain[-3:]=='.mz': s='mozambique'
    if sDomain[-3:]=='.mm': s='myanmar'
    if sDomain[-3:]=='.na': s='namibia'
    if sDomain[-3:]=='.nr': s='nauru'
    if sDomain[-3:]=='.np': s='nepal'
    if sDomain[-3:]=='.nl': s='netherlands'
    if sDomain[-3:]=='.nc': s='new caledonia (french overseas collectivity)'
    if sDomain[-3:]=='.nz': s='new zealand'
    if sDomain[-3:]=='.ni': s='nicaragua'
    if sDomain[-3:]=='.ne': s='niger'
    if sDomain[-3:]=='.ng': s='nigeria'
    if sDomain[-3:]=='.nu': s='niue'
    if sDomain[-3:]=='.nf': s='norfolk island'
    if sDomain[-3:]=='.kp (stands for democratic people’s republic of korea)': s='north korea'
    if sDomain[-3:]=='.mk (stands for македонија or, in the latin alphabet, makedonija)': s='north macedonia'
    if sDomain[-3:]=='.uk': s='northern ireland'
    if sDomain[-3:]=='.mp': s='northern mariana islands'
    if sDomain[-3:]=='.no': s='norway'
    if sDomain[-3:]=='.om': s='oman'
    if sDomain[-3:]=='.pk': s='pakistan'
    if sDomain[-3:]=='.pw (stands for pelew)': s='palau'
    if sDomain[-3:]=='.ps': s='palestine'
    if sDomain[-3:]=='.pa': s='panama'
    if sDomain[-3:]=='.pg': s='papua new guinea'
    if sDomain[-3:]=='.py': s='paraguay'
    if sDomain[-3:]=='.pe': s='peru'
    if sDomain[-3:]=='.ph': s='philippines'
    if sDomain[-3:]=='.pn': s='pitcairn islands'
    if sDomain[-3:]=='.pl': s='poland'
    if sDomain[-3:]=='.pt': s='portugal'
    if sDomain[-3:]=='.pr': s='puerto rico'
    if sDomain[-3:]=='.qa': s='qatar'
    if sDomain[-3:]=='.re': s='réunion (french overseas department)'
    if sDomain[-3:]=='.ro': s='romania'
    if sDomain[-3:]=='.ru': s='russia'
    if sDomain[-3:]=='.rw': s='rwanda'
    if sDomain[-3:]=='.sh': s='saint helena'
    if sDomain[-3:]=='.kn': s='saint kitts and nevis'
    if sDomain[-3:]=='.lc': s='saint lucia'
    if sDomain[-3:]=='.mf':  s='saint martin (french overseas collectivity)'
    if sDomain[-3:]=='.vc': s='saint vincent and the grenadines'
    if sDomain[-3:]=='.pm': s='saint-pierre and miquelon (french overseas collectivity)'
    if sDomain[-3:]=='.ws': s='samoa'
    if sDomain[-3:]=='.sm': s='san marino'
    if sDomain[-3:]=='.st': s='são tomé and príncipe'
    if sDomain[-3:]=='.sa': s='saudi arabia'
    if sDomain[-3:]=='.uk': s='scotland'
    if sDomain[-3:]=='.sn': s='senegal'
    if sDomain[-3:]=='.rs': s='serbia'
    if sDomain[-3:]=='.sc': s='seychelles'
    if sDomain[-3:]=='.sl': s='sierra leone'
    if sDomain[-3:]=='.sg': s='singapore'
    if sDomain[-3:]=='.an': s='sint eustatius'
    if sDomain[-3:]=='.sx': s='sint maarten'
    if sDomain[-3:]=='.sk': s='slovakia'
    if sDomain[-3:]=='.si': s='slovenia'
    if sDomain[-3:]=='.sb': s='solomon islands'
    if sDomain[-3:]=='.so': s='somalia'
    if sDomain[-3:]=='.so': s='somaliland (unrecognised, self-declared state)'
    if sDomain[-3:]=='.za': s='south africa'
    if sDomain[-3:]=='.gs': s='south georgia and the south sandwich islands'
    if sDomain[-3:]=='.kr': s='south korea'
    if sDomain[-3:]=='.ss': s='south sudan'
    if sDomain[-3:]=='.es': s='spain'
    if sDomain[-3:]=='.lk': s='sri lanka'
    if sDomain[-3:]=='.sd': s='sudan'
    if sDomain[-3:]=='.sr': s='suriname (surinam)'
    if sDomain[-3:]=='.sj': s='svalbard and jan mayen islands'
    if sDomain[-3:]=='.sz': s='swaziland'
    if sDomain[-3:]=='.se': s='sweden'
    if sDomain[-3:]=='.ch': s='switzerland'
    if sDomain[-3:]=='.sy': s='syria'
    if sDomain[-3:]=='.pf': s='tahiti'
    if sDomain[-3:]=='.tw': s='taiwan'
    if sDomain[-3:]=='.tj': s='tajikistan'
    if sDomain[-3:]=='.tz': s='tanzania'
    if sDomain[-3:]=='.th': s='thailand'
    if sDomain[-3:]=='.tg': s='togo'
    if sDomain[-3:]=='.tk': s='tokelau'
    if sDomain[-3:]=='.to': s='tonga'
    if sDomain[-3:]=='.tt': s='trinidad and tobago'
    if sDomain[-3:]=='.tn': s='tunisia'
    if sDomain[-3:]=='.tr': s='turkey'
    if sDomain[-3:]=='.tm': s='turkmenistan'
    if sDomain[-3:]=='.tc': s='turks and caicos islands'
    if sDomain[-3:]=='.tv': s='tuvalu'
    if sDomain[-3:]=='.ug': s='uganda'
    if sDomain[-3:]=='.ua': s='ukraine'
    if sDomain[-3:]=='.ae': s='united arab emirates (uae)'
    if sDomain[-3:]=='.uk': s='united kingdom (uk)'
    if sDomain[-3:]=='.us': s='united states of america (usa)'
    if sDomain[-3:]=='.vi': s='united states virgin islands'
    if sDomain[-3:]=='.uy': s='uruguay'
    if sDomain[-3:]=='.uz': s='uzbekistan'
    if sDomain[-3:]=='.vu': s='vanuatu'
    if sDomain[-3:]=='.va': s='vatican city'
    if sDomain[-3:]=='.ve': s='venezuela'
    if sDomain[-3:]=='.vn': s='vietnam'
    if sDomain[-3:]=='.uk': s='wales'
    if sDomain[-3:]=='.wf': s='wallis and futuna (french overseas collectivity)'
    if sDomain[-3:]=='.ps': s='west bank'
    if sDomain[-3:]=='.eh': s='western sahara'
    if sDomain[-3:]=='.ye': s='yemen'
    if sDomain[-3:]=='.zm': s='zambia'
    if sDomain[-3:]=='.zw': s='zimbabwe'
    if sDomain[-3:]=='com': s='united states of america (usa)'
    if sDomain[-3:]=='org': s='united states of america (usa)'
    if sDomain[-3:]=='gov': s='united states of america (usa)'
    if sDomain[-3:]=='mil': s='united states of america (usa)'
    if sDomain[-3:]=='net': s='united states of america (usa)'
    if sDomain[-3:]=='edu': s='united states of america (usa)'
    return s;


def BinAllocate(row):
    if 0 <= row['PwnCount'] <= 154580998:
           s=1
    if 154580999 <= row['PwnCount'] <= 309161996:
           s=2
    if 309161997 <= row['PwnCount'] <= 463742994:
           s=3
    if 463742995 <= row['PwnCount'] <= 618323992:
           s=4
    if 618323993 <= row['PwnCount'] <= 772904991:
           s=5
    return s;


df= pd.read_csv('databreachescsv.csv')



df['country'] = df.apply(lambda row: WebDomain2Country(row), axis=1)
df['bin'] = df.apply(lambda row: BinAllocate(row), axis=1)

#df['country_id'] = pd.factorize(df['country'])[0]
#df.drop(columns=['country'], axis = 1)
#df.drop(columns = ['country'])
#del df['country']


df.to_csv('databreachesThi.csv')