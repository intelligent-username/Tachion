# Frontend

this contains all of the code required for the front end. I will keep it minimal, and finish it last.

I will try to restrict this folder to only the following files:

- `index.html`: for the main page
- `styles.css`: for the main styles
- `main.js`: wires all of the front end logic together (import, bootstrap, initial state)

"Logical" operations:

```nd
js/
  api.js:         # calls the API, retrieves formatted data
  ui.js:          # DOM mutations
  events.js:      # DOM listeners
  visualizer.js:  # preps data for graphing
  state.js:       # holds in-memory data for other sources
```

Visual components:

```md
components/
  sidebar.js:     # sidebar: search feature, settings, details
  header.js:      # header: app name, logo, current date
  footer.js:      # footer: links to GitHub, documentation, etc.
  graph.js:       # prepared data from visualizer.js -> plots
```
