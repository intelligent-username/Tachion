// Event handler utilities

// Handle keyboard submit (Enter key)
export const handleKeyboardSubmit = (e, onSubmit) => {
    if (e.key === 'Enter') {
        onSubmit()
    }
}
