// Wire the app together

const componentModules = [
	'./components/footer.js',
	'./components/graph.js',
	'./components/header.js',
	'./components/sidebar.js',
];
const jsModules = [
	'./js/api.js',
	'./js/events.js',
	'./js/state.js',
	'./js/ui.js',
	'./js/visualizer.js',
];

[...componentModules, ...jsModules].forEach((path) => {
	import(path).catch((err) => {
		console.error(`Failed to load module: ${path}`, err);
	});
});


// Display everything

// Make sure to also give an option to clearly print/display the predictions, errors, and whatever other options are implemented for the user
