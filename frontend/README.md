# Frontend

this contains all of the code required for the front end. I will keep it minimal, and finish it last.

I will try to restrict this folder to only the following files:

- `index.html`: for the main page
- `styles.css`: for the main styles
- `main.js`: wires all of the front end logic together

- `js/`: JavaScript files that are for "logical" operations
  - `api.js`: calls the API, retrieves formatted data
  - `ui.js`: UI updates & interactions
  - `events.js`: event listeners
  - `visualizer.js`: takes in properly formatted data and graphs using D3

- `components/`: all visual components
  - `sidebar.js`: sidebar, contains search feature, settings, & details
  - `header.js`: just show the app name, logo, current, date
  - `footer.js`: footer with links to GitHub, documentation, etc.
  - `graph.js`: graph components, get details from visualizer.js

