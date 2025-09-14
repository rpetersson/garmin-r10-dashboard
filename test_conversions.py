#!/usr/bin/env python3
"""Test unit conversion functions for the Garmin R10 Dashboard"""

def test_conversions():
    # Test conversion functions from app.py
    
    def convert_speed_to_usa(kmh_value):
        """Convert km/h to mph"""
        if kmh_value is None:
            return kmh_value
        return kmh_value * 0.621371

    def convert_distance_to_usa(meter_value):
        """Convert meters to yards"""
        if meter_value is None:
            return meter_value
        return meter_value * 1.09361

    # Test speed conversion
    print("Speed Conversions (km/h to mph):")
    test_speeds = [100, 150, 160, 180, 200]
    for kmh in test_speeds:
        mph = convert_speed_to_usa(kmh)
        print(f"  {kmh} km/h = {mph:.1f} mph")
    
    print("\nDistance Conversions (meters to yards):")
    test_distances = [100, 150, 200, 250, 300]
    for meters in test_distances:
        yards = convert_distance_to_usa(meters)
        print(f"  {meters}m = {yards:.1f} yards")
    
    # Verify some known conversions
    print("\nVerification:")
    print(f"  100 km/h = {convert_speed_to_usa(100):.2f} mph (should be ~62.14)")
    print(f"  100m = {convert_distance_to_usa(100):.2f} yards (should be ~109.36)")

if __name__ == "__main__":
    test_conversions()