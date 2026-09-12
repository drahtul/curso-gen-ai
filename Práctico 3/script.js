const themeToggle = document.getElementById("theme-toggle");
const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
const root = document.documentElement;

const setTheme = (theme) => {
    if (theme === "dark") {
        root.dataset.theme = "dark";
        themeToggle.textContent = "Modo claro";
    } else {
        root.dataset.theme = "light";
        themeToggle.textContent = "Modo oscuro";
    }
    localStorage.setItem("site-theme", theme);
};

const savedTheme = localStorage.getItem("site-theme");
if (savedTheme) {
    setTheme(savedTheme);
} else {
    setTheme(prefersDark ? "dark" : "light");
}

themeToggle.addEventListener("click", () => {
    const currentTheme = root.dataset.theme === "dark" ? "light" : "dark";
    setTheme(currentTheme);
});
