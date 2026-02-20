// Header component
import { formatDate } from '../js/ui'

export default function Header() {
    const today = new Date()

    return (
        <header className="header">
            <div className="brand">
                <img src="/Logo.svg" alt="Tachion Logo" className="logo" />
                <h1>Tachion</h1>
            </div>
            <span className="date">{formatDate(today)}</span>
        </header>
    )
}
