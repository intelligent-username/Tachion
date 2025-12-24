# Frontend

this contains all of the code required for the front end. I will keep it minimal, and finish it last.

I will try to restrict this folder to only the following files:

- `index.html`: for the main page
- `styles.css`: for the main styles
- `main.js`: wires all of the front end logic together (import, bootstrap, initial state)

- `js/`: "Logical" operations
  - `api.js`: calls the API, retrieves formatted data
  - `ui.js`: DOM mutations
  - `events.js`: DOM listeners
  - `visualizer.js`: takes in properly formatted data and prepares for graphing (scale, etc.)
  - `state.js`: holds the current in-memory data (for other components to reference/update); a centralized source

- `components/`: visual components
  - `sidebar.js`: sidebar, contains search feature, settings, & details
  - `header.js`: just show the app name, logo, current, date
  - `footer.js`: footer with links to GitHub, documentation, etc.
  - `graph.js`: takes in prepared data (from `visualizer.js`) renders using D3. <!-- Nothing else!! -->

