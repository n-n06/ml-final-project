import os
from dotenv import load_dotenv

from opensky_api import OpenSkyApi

load_dotenv()


def main():
    CLIENT_ID = os.environ["CLIENT_ID"]
    CLIENT_SECRET = os.environ["CLIENT_SECRET"]

    api = OpenSkyApi()
    arrivals = api.get_arrivals_by_airport("EDDF", 1517227200, 1517230800)
    departures = api.get_departures_by_airport("EDDF", 1517227200, 1517230800)
    print("Arrivals:")
    for flight in arrivals:
        print(flight)
    print("Departures:")
    for flight in departures:
        print(flight)
    


if __name__ == "__main__":
    main()
