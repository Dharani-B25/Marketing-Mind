import pandas as pd
import numpy as np

# Same results every time
np.random.seed(42)

# Dates
dates = pd.date_range(
    start="2026-01-01",
    end="2026-06-30",
    freq="D"
)

# Campaigns
campaigns = [
    "Summer Sale",
    "Product Launch",
    "Brand Awareness",
    "Retargeting",
    "Festival Offer"
]

# Marketing channels
channels = [
    "Google Ads",
    "Facebook Ads",
    "Instagram Ads",
    "LinkedIn Ads",
    "YouTube Ads"
]

data = []

# Generate marketing data
for date in dates:

    for campaign in campaigns:

        channel = np.random.choice(channels)

        # Number of people who saw the advertisement
        impressions = np.random.randint(5000, 50000)

        # Click percentage
        ctr = np.random.uniform(0.01, 0.08)

        # Number of clicks
        clicks = int(impressions * ctr)

        # Conversion percentage
        conversion_rate = np.random.uniform(0.02, 0.15)

        # Number of conversions
        conversions = int(clicks * conversion_rate)

        # Advertising cost
        spend = round(np.random.uniform(100, 2000), 2)

        # Revenue generated per conversion
        revenue_per_conversion = np.random.uniform(500, 3000)

        # Total revenue
        revenue = round(
            conversions * revenue_per_conversion,
            2
        )

        data.append([
            date,
            campaign,
            channel,
            impressions,
            clicks,
            conversions,
            spend,
            revenue
        ])


# Convert data into DataFrame
df = pd.DataFrame(
    data,
    columns=[
        "Date",
        "Campaign",
        "Channel",
        "Impressions",
        "Clicks",
        "Conversions",
        "Spend",
        "Revenue"
    ]
)

# Save CSV file
df.to_csv(
    "data/marketing_data.csv",
    index=False
)

# Display result
print("====================================")
print("MarketMind AI Dataset Created!")
print("====================================")

print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset saved to:")
print("data/marketing_data.csv")