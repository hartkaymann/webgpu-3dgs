export class Utils {

    static showToast(message: string, type: 'error' | 'warn' | 'info'= 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        const container = document.getElementById('toast-container');
        container?.appendChild(toast);

        setTimeout(() => toast.remove(), 5000);
    }
}