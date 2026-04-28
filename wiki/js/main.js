/* ml-math-foundations wiki — main.js */

// ── Global nav (edit here to update all pages) ────────────────────────────────
function buildNav() {
  const nav = document.querySelector('#sidebar nav');
  if (!nav) return;

  // Determine path prefix based on whether we're in a subdirectory
  const inSubdir = window.location.pathname.includes('/linear-algebra/');
  const home = inSubdir ? '../index.html' : 'index.html';
  const la   = inSubdir ? '' : 'linear-algebra/';

  nav.innerHTML = `
    <ul>
      <li><a href="${home}">Home</a></li>
    </ul>

    <div class="nav-section">Linear Algebra</div>
    <ul>
      <li><a href="${la}vectors.html">Lesson 1: Vectors</a></li>
      <li><a href="${la}span.html">Lesson 2: Span</a></li>
      <li><a href="${la}matrices.html">Lesson 3: Matrices</a></li>
      <li><a href="${la}matrix-multiplication.html">Lesson 4: Matrix Multiplication</a></li>
      <li><a href="${la}3d-transformations.html">Lesson 5: 3D Transformations</a></li>
      <li><a href="${la}determinants.html">Lesson 6: Determinants</a></li>
      <li><a href="${la}inverse-column-null.html">Lesson 7: Inverse, Column Space, Null Space</a></li>
      <li><a href="${la}nonsquare-matrices.html">Lesson 8: Non-Square Matrices</a></li>
    </ul>

    <div class="nav-section">Calculus</div>
    <ul>
      <li><a class="coming-soon">Coming soon</a></li>
    </ul>

    <div class="nav-section">Probability &amp; Stats</div>
    <ul>
      <li><a class="coming-soon">Coming soon</a></li>
    </ul>
  `;
}

buildNav();

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
