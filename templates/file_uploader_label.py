hide_label = (
    """
<style>
    div[data-testid="stFileUploader"]>section[data-testid="stFileUploadDropzone"]>button[data-testid="baseButton-secondary"] {
       color:white;
    }
    div[data-testid="stFileUploader"]>section[data-testid="stFileUploadDropzone"]>button[data-testid="baseButton-secondary"]::after {
        content: "Buscar";
        color:black;
        display: block;
        position: absolute;
    }
    div[data-testid="stFileDropzoneInstructions"]>div>span {
       visibility:hidden;
    }
    div[data-testid="stFileDropzoneInstructions"]>div>span::after {
       content:"Arraste os arquivos aqui";
       visibility:visible;
       display:block;
    }
     div[data-testid="stFileDropzoneInstructions"]>div>small {
       visibility:hidden;
    }
    div[data-testid="stFileDropzoneInstructions"]>div>small::before {
       content:"Limite de 200MB por aquivo";
       visibility:visible;
       display:block;
    }
</style>
"""
)