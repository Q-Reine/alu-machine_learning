#!/usr/bin/env python3
"""Retrieves ships that can hold a given number of passengers."""
import requests


def availableShips(passengerCount):
    """Returns list of ships that can hold given number of passengers."""
    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data['results']:
            passengers = ship['passengers'].replace(',', '')
            if passengers.isdigit() and int(passengers) >= passengerCount:
                ships.append(ship['name'])

        url = data['next']

    return ships
