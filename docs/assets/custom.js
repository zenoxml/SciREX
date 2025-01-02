// Redirect the logo to SciREX.org without refreshing the page
document.addEventListener('DOMContentLoaded', function () {
    const logo = document.querySelector('.md-header__button.md-logo');
    if (logo) {
        logo.addEventListener('click', function (event) {
            event.preventDefault(); 
            window.location.href = 'https://scirex.org'; 
        });
    }
});
