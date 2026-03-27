// Main JavaScript file for Pest Identifier

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize all functions
    initFileUpload();
    initBackToTop();
    initTooltips();
    initSearch();
    
});

// File Upload Handling
function initFileUpload() {
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = this.files[0];
            if (file) {
                validateAndPreview(file);
            }
        });
    }
}

// Validate and preview uploaded image
function validateAndPreview(file) {
    const fileSize = file.size / 1024 / 1024; // Convert to MB
    const fileType = file.type;
    
    // Check file size (max 16MB)
    if (fileSize > 16) {
        showAlert('File size must be less than 16MB', 'danger');
        document.getElementById('file').value = '';
        return false;
    }
    
    // Check file type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/jpg'];
    if (!allowedTypes.includes(fileType)) {
        showAlert('Please upload an image file (JPEG, PNG, GIF)', 'danger');
        document.getElementById('file').value = '';
        return false;
    }
    
    // Show image preview
    previewImage(file);
    return true;
}

// Preview image before upload
function previewImage(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.getElementById('preview-image');
        const previewDiv = document.getElementById('preview');
        
        if (preview && previewDiv) {
            preview.src = e.target.result;
            previewDiv.classList.remove('d-none');
            
            // Add animation
            preview.style.animation = 'fadeIn 0.5s';
        }
    }
    reader.readAsDataURL(file);
}

// Show alert messages
function showAlert(message, type) {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of main content
    const main = document.querySelector('main');
    if (main) {
        main.insertBefore(alertDiv, main.firstChild);
        
        // Auto dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
}

// Copy text to clipboard
function copyToClipboard(text, elementId) {
    navigator.clipboard.writeText(text).then(function() {
        showAlert('Copied to clipboard!', 'success');
        
        // Highlight the copied element
        const element = document.getElementById(elementId);
        if (element) {
            element.style.backgroundColor = '#d4edda';
            setTimeout(() => {
                element.style.backgroundColor = '';
            }, 1000);
        }
    }, function(err) {
        showAlert('Failed to copy text', 'danger');
        console.error('Could not copy text: ', err);
    });
}

// Search functionality
function initSearch() {
    const searchInput = document.getElementById('pestSearch');
    if (searchInput) {
        searchInput.addEventListener('keyup', function() {
            searchPests(this.value);
        });
    }
}

function searchPests(searchTerm) {
    const cards = document.getElementsByClassName('pest-card');
    const term = searchTerm.toUpperCase();
    
    for (let i = 0; i < cards.length; i++) {
        const title = cards[i].getElementsByClassName('card-title')[0];
        const description = cards[i].getElementsByClassName('card-text')[0];
        
        if (title && description) {
            const titleText = title.textContent || title.innerText;
            const descText = description.textContent || description.innerText;
            
            if (titleText.toUpperCase().indexOf(term) > -1 || 
                descText.toUpperCase().indexOf(term) > -1) {
                cards[i].style.display = '';
                cards[i].style.animation = 'fadeIn 0.5s';
            } else {
                cards[i].style.display = 'none';
            }
        }
    }
}

// Back to Top Button
function initBackToTop() {
    // Create button if it doesn't exist
    if (!document.getElementById('back-to-top')) {
        const btn = document.createElement('button');
        btn.id = 'back-to-top';
        btn.innerHTML = '↑';
        btn.title = 'Back to Top';
        document.body.appendChild(btn);
    }
    
    const backToTopBtn = document.getElementById('back-to-top');
    
    window.addEventListener('scroll', function() {
        if (window.pageYOffset > 300) {
            backToTopBtn.classList.add('show');
        } else {
            backToTopBtn.classList.remove('show');
        }
    });
    
    backToTopBtn.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// Initialize Bootstrap tooltips
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Show loading spinner
function showLoading() {
    const loadingSpinner = document.getElementById('loading-spinner');
    if (loadingSpinner) {
        loadingSpinner.classList.remove('d-none');
    } else {
        // Create spinner if it doesn't exist
        const spinner = document.createElement('div');
        spinner.id = 'loading-spinner';
        spinner.className = 'text-center my-5';
        spinner.innerHTML = '<div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div>';
        document.querySelector('main').appendChild(spinner);
    }
}

// Hide loading spinner
function hideLoading() {
    const loadingSpinner = document.getElementById('loading-spinner');
    if (loadingSpinner) {
        loadingSpinner.classList.add('d-none');
    }
}

// Filter pests by type
function filterPests(type) {
    const cards = document.getElementsByClassName('pest-card');
    const buttons = document.getElementsByClassName('filter-btn');
    
    // Update active button
    for (let i = 0; i < buttons.length; i++) {
        buttons[i].classList.remove('active');
    }
    event.target.classList.add('active');
    
    // Filter cards
    for (let i = 0; i < cards.length; i++) {
        if (type === 'all' || cards[i].dataset.type === type) {
            cards[i].style.display = '';
        } else {
            cards[i].style.display = 'none';
        }
    }
}

// Print treatment information
function printTreatment(pestName) {
    const printContent = document.getElementById('treatment-info');
    if (printContent) {
        const originalTitle = document.title;
        document.title = `Treatment for ${pestName}`;
        
        const printWindow = window.open('', '_blank');
        printWindow.document.write(`
            <html>
                <head>
                    <title>Treatment for ${pestName}</title>
                    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                    <link rel="stylesheet" href="/static/css/style.css">
                </head>
                <body>
                    <div class="container mt-4">
                        ${printContent.innerHTML}
                    </div>
                </body>
            </html>
        `);
        
        printWindow.document.close();
        printWindow.focus();
        printWindow.print();
        printWindow.close();
        
        document.title = originalTitle;
    }
}

// Add animation class to elements when they come into view
function observeElements() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fadeIn');
            }
        });
    });
    
    document.querySelectorAll('.card, .alert, .stat-card').forEach(el => {
        observer.observe(el);
    });
}

// Call observeElements if IntersectionObserver is supported
if ('IntersectionObserver' in window) {
    observeElements();
}

// Add CSS animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fadeIn {
        animation: fadeIn 0.5s ease forwards;
    }
`;
document.head.appendChild(style);