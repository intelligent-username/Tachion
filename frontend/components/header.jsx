// Header component
import { formatDate } from '../js/ui'

export default function Header() {
    const today = new Date()

    return (
        <header className="header">
            <img src="public/Logo.svg" alt="Tachion Logo" className="logo" />
            <h1>Tachion</h1>
            <span className="date">{formatDate(today)}</span>
        </header>
    )
}
