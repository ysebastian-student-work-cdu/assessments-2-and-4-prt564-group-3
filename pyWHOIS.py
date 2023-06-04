import python-whois
 
def is_registered(domain_name):
    """
    A function that returns a boolean indicating
    whether a `domain_name` is registered
    """
    try:
        w = whois.whois(domain_name)
    except Exception:
        return False
    else:
        return bool(w.domain_name)
domains = [
    "thepythoncode.com",
    "google.com",
    "github.com",
    "unknownrandomdomain.com",
    "notregistered.co"
]
# iterate over domains



for domain in domains:
    whois_info = whois.whois(domain)
    print(domain)
    print(  is_registered(domain) )
    print("registrar", whois_info.registrar)
    print("WHOIS server:", whois_info.whois_server)