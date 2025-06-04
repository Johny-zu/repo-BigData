# Dashboard Panel

## Overview
This project implements a credit risk dashboard using Panel, which allows users to analyze credit information of borrowers and predict potential defaults. The dashboard provides various filters and visualizations to help users make informed financial decisions.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd dashboard-panel
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Dashboard

To run the dashboard locally, execute the following command:
```
python src/dashboard.py
```

Once the server is running, open your web browser and navigate to `http://localhost:5006` to view the dashboard.

## Usage

- Use the filters on the left side to adjust the parameters for analysis.
- The dashboard will display various visualizations based on the selected filters.
- You can view aggregated data in the table section at the bottom.

## Dependencies

The project requires the following Python packages:
- Panel
- Pandas
- NumPy
- Holoviews
- hvplot

