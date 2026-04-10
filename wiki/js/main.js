/* ml-math-foundations wiki — main.js */

// Mobile sidebar toggle
const menuBtn = document.getElementById('menu-btn');
const sidebar = document.getElementById('sidebar');

if (menuBtn && sidebar) {
  menuBtn.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });

  // Close sidebar when clicking outside
  document.addEventListener('click', (e) => {
    if (!sidebar.contains(e.target) && !menuBtn.contains(e.target)) {
      sidebar.classList.remove('open');
    }
  });
}

// Mark current nav link as active based on path
document.querySelectorAll('.sidebar nav a').forEach(link => {
  const href = link.getAttribute('href');
  if (!href || href === '#') return;

  // Resolve the link href relative to current page
  const linkUrl = new URL(href, window.location.href);
  if (linkUrl.pathname === window.location.pathname) {
    link.classList.add('active');
  }
});
