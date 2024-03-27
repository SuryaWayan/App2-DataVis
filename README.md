# data-empower

Develop a Python-based web application using Streamlit tailored specifically for data analysts, aiming to effortlessly visualize and analyze CSV data. Ensure the application delivers a seamless user experience with clear instructions provided at each step. The primary objective is to empower users to explore and comprehend their data effectively.

Page 1:

CSV Upload:

Implement a seamless CSV file upload button.
Upon upload, display a confirmation message and allow users to proceed.
Data Overview:

Provide a summary of the uploaded CSV data, indicating total rows and columns.
Offer users an initial understanding of the dataset's size and structure.
Column List:

Present a comprehensive list of all column names in the uploaded dataset.
Assist users in identifying and selecting specific columns for further analysis.
Interactive Table Generation:

Allow users to select desired columns and specify the number of rows before displaying the table.
Enhance user control for focused data exploration.
All user input or button related to each chart should be located to the left of the chart for consistency.




Page 2:

Dynamic Chart Generation:
Offer a variety of visually appealing charts for data visualization.
Allow configuration of up to 10 different charts vertically.
Enable users to specify X and Y axes from available dataset columns.
Provide an option for the user to change the X-axis data type to either date or number for each chart.
Move the chart summary to the right of each chart for better visibility.
Support multiple columns for the Y axis to facilitate comprehensive data comparison.
Support secondary Y axis if required, giving users the option to choose.
Gracefully handle absence of data when no configuration is specified.
Provide options for adding trend lines to EACH chart (e.g., linear, exponential, polynomial) and display the function and R VALUES of the trend lines.
Allow users to customize chart appearance, including colors, labels, and titles, enhancing flexibility to suit individual preferences.
Allow users to input the start and end of the X axis, with the default setting showing the whole X axis range.

note: all the user input or button related for each chart should be located to the left of the chart.

Both in Page 1 and 2:

Interactive User Guide:

Incorporate an interactive user guide or tutorial within the application.
Provide step-by-step instructions, tooltips, and examples to enhance user proficiency and confidence.
Responsive Design:

Ensure responsiveness across various devices and screen sizes.
Deliver a consistent and optimized user experience regardless of the platform used.


Note:
Users can navigate between Page 1 and Page 2 without needing to re-upload the CSV.



By integrating these features in the specified sequence, the Python-based web application will empower data analysts to efficiently explore, visualize, and derive insights from their CSV data.

## Collaborate with GPT Engineer

This is a [gptengineer.app](https://gptengineer.app)-synced repository 🌟🤖

Changes made via gptengineer.app will be committed to this repo.

If you clone this repo and push changes, you will have them reflected in the GPT Engineer UI.

## Setup

```sh
git clone https://github.com/GPT-Engineer-App/data-empower.git
cd data-empower
npm i
```

```sh
npm run dev
```

This will run a dev server with auto reloading and an instant preview.

## Tech stack

- [Vite](https://vitejs.dev/)
- [React](https://react.dev/)
- [Chakra UI](https://chakra-ui.com/)

## Requirements

- Node.js & npm - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)
