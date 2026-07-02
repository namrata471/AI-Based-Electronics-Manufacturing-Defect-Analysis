import pandas as pd
import random
from datetime import datetime, timedelta

products = [
    "PCB", "Battery Pack", "Sensor Module", "Camera Module",
    "Display Unit", "Microcontroller Board", "Relay Module",
    "Power Supply Unit", "Wi-Fi Module", "LED Driver"
]

machines = ["M1", "M2", "M3", "M4", "M5"]

production_lines = ["L1", "L2", "L3", "L4"]

shifts = ["Morning", "Evening", "Night"]

operators = [f"OP{100+i}" for i in range(1, 31)]

materials = ["FR4", "Ceramic", "Copper", "Aluminum"]

defects = [
    "Missing Component",
    "Solder Bridge",
    "Short Circuit",
    "Open Circuit",
    "PCB Crack",
    "Misalignment",
    "Burn Mark",
    "Loose Connection"
]

maintenance = ["Good", "Warning", "Critical"]

severity_levels = ["Low", "Medium", "High"]

inspection_result = ["Pass", "Fail"]

data = []

start_date = datetime(2025,1,1)

for i in range(1,1001):

    temperature = random.randint(35,55)
    humidity = random.randint(40,65)
    voltage = round(random.uniform(3.3,230),2)
    current = round(random.uniform(0.2,5.0),2)
    vibration = round(random.uniform(0.5,4.5),2)
    speed = random.randint(80,150)
    downtime = random.randint(0,60)
    risk = random.randint(10,100)

    severity = random.choices(
        severity_levels,
        weights=[50,30,20]
    )[0]

    data.append([
        f"I{i:04}",
        random.choice(products),
        random.choice(production_lines),
        random.choice(machines),
        random.choice(shifts),
        temperature,
        humidity,
        voltage,
        current,
        vibration,
        speed,
        random.choice(operators),
        random.choice(materials),
        random.choice(defects),
        severity,
        random.choice(inspection_result),
        downtime,
        random.choice(maintenance),
        risk,
        (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    ])

columns = [
    "Inspection_ID",
    "Product",
    "Production_Line",
    "Machine",
    "Shift",
    "Temperature",
    "Humidity",
    "Voltage",
    "Current",
    "Vibration",
    "Production_Speed",
    "Operator",
    "Material_Type",
    "Defect_Type",
    "Severity",
    "Inspection_Result",
    "Downtime_Minutes",
    "Maintenance_Status",
    "AI_Risk_Score",
    "Inspection_Date"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("data/manufacturing_defects.csv", index=False)

print("✅ Dataset with 1000 records created successfully!")